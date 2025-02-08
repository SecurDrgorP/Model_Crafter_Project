import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count, Manager
import gc
from config import MODEL_DIR, logger

def process_batch(batch_data, chunk_queue):
    """Process a single batch of data and add to shared queue"""
    try:
        images, labels = batch_data
        # Ensure consistent shape by flattening properly
        processed_x = images.reshape(images.shape[0], -1).astype(np.float32)
        processed_y = np.argmax(labels, axis=1) if len(labels.shape) > 1 else labels
        chunk_queue.put((processed_x, processed_y))
        del images, labels, processed_x, processed_y
        gc.collect()
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        logger.error(f"Input shapes - images: {images.shape}, labels: {labels.shape}")
        raise

def process_generator(generator, chunk_size=1000, n_jobs=-1):
    """Process entire generator and return data in chunks"""
    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)

    total_batches = len(generator)
    processed_batches = 0
    
    chunks_x = []
    chunks_y = []
    current_chunk_x = []
    current_chunk_y = []
    
    # Store first batch shapes for validation
    first_batch = next(generator)
    first_batch_shape = first_batch[0][0].shape
    processed_batches += 1
    
    # Create a managed queue for inter-process communication
    with Manager() as manager:
        chunk_queue = manager.Queue()
        
        # Process first batch
        process_batch(first_batch, chunk_queue)
        
        # Process remaining batches with limited parallel workers
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            while processed_batches < total_batches:
                try:
                    batch = next(generator)
                    # Validate batch shape
                    if batch[0][0].shape != first_batch_shape:
                        raise ValueError(f"Inconsistent batch shape. Expected {first_batch_shape}, got {batch[0][0].shape}")
                    
                    executor.submit(process_batch, batch, chunk_queue)
                    processed_batches += 1
                    
                    # Process queue
                    while not chunk_queue.empty():
                        batch_x, batch_y = chunk_queue.get()
                        current_chunk_x.append(batch_x)
                        current_chunk_y.append(batch_y)
                        
                        # If chunk is full, store it
                        if sum(len(x) for x in current_chunk_x) >= chunk_size:
                            try:
                                X_chunk = np.vstack(current_chunk_x)
                                y_chunk = np.concatenate(current_chunk_y)
                                chunks_x.append(X_chunk)
                                chunks_y.append(y_chunk)
                                logger.info(f"Created chunk of shape {X_chunk.shape}")
                            except ValueError as e:
                                logger.error(f"Error creating chunk: {str(e)}")
                                logger.error(f"Chunk shapes - X: {[x.shape for x in current_chunk_x]}, y: {[y.shape for y in current_chunk_y]}")
                                raise
                            finally:
                                del X_chunk, y_chunk, current_chunk_x, current_chunk_y
                                current_chunk_x = []
                                current_chunk_y = []
                                gc.collect()
                    
                    if processed_batches % (total_batches // 10) == 0:
                        logger.info(f"Processed {processed_batches}/{total_batches} batches")
                
                except StopIteration:
                    break
                except Exception as e:
                    logger.error(f"Error processing batch {processed_batches}: {str(e)}")
                    raise
        
        # Process remaining data in queue
        while not chunk_queue.empty():
            batch_x, batch_y = chunk_queue.get()
            current_chunk_x.append(batch_x)
            current_chunk_y.append(batch_y)
        
        # Add final chunk if any data remains
        if current_chunk_x:
            try:
                X_chunk = np.vstack(current_chunk_x)
                y_chunk = np.concatenate(current_chunk_y)
                chunks_x.append(X_chunk)
                chunks_y.append(y_chunk)
                logger.info(f"Created final chunk of shape {X_chunk.shape}")
            except ValueError as e:
                logger.error(f"Error creating final chunk: {str(e)}")
                logger.error(f"Final chunk shapes - X: {[x.shape for x in current_chunk_x]}, y: {[y.shape for y in current_chunk_y]}")
                raise
            finally:
                del X_chunk, y_chunk, current_chunk_x, current_chunk_y
                gc.collect()
    
    return chunks_x, chunks_y

def train_traditional_models(train_gen, test_gen, chunk_size=4000, n_jobs=-1):
    """Train models using chunked data processing"""
    try:
        if n_jobs == -1:
            n_jobs = max(1, cpu_count() - 4)
        
        logger.info(f"Using {n_jobs} CPU cores")
        
        # Process training data
        logger.info("Processing training data...")
        train_chunks_x, train_chunks_y = process_generator(
            train_gen,
            chunk_size=chunk_size,
            n_jobs=n_jobs
        )
        
        # Validate chunk shapes
        x_shapes = [x.shape for x in train_chunks_x]
        y_shapes = [y.shape for y in train_chunks_y]
        logger.info(f"Training chunk shapes - X: {x_shapes}, y: {y_shapes}")
        
        # Initialize preprocessor
        preprocessor = make_pipeline(
            StandardScaler(with_mean=False),
            TruncatedSVD(n_components=100, random_state=42)
        )
        
        # Fit preprocessor on first chunk
        logger.info("Fitting preprocessor...")
        X_first_chunk = preprocessor.fit_transform(train_chunks_x[0])
        
        # Train models
        logger.info("Training models...")
        models = {
            'SVM': SVC(kernel='rbf', probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'RandomForest': RandomForestClassifier(n_estimators=100, n_jobs=1)
        }
        
        # Train on chunks
        for i, (X_chunk, y_chunk) in enumerate(zip(train_chunks_x, train_chunks_y)):
            # Preprocess chunk
            X_chunk = preprocessor.transform(X_chunk) if i > 0 else X_first_chunk
            
            # Train models on chunk
            for name, model in models.items():
                try:
                    if i == 0:
                        model.fit(X_chunk, y_chunk)
                    else:
                        # For models that don't support partial_fit, we retrain
                        model.fit(X_chunk, y_chunk)
                except Exception as e:
                    logger.error(f"Error training {name} on chunk {i}: {str(e)}")
                    logger.error(f"Chunk shapes - X: {X_chunk.shape}, y: {y_chunk.shape}")
                    raise
            
            logger.info(f"Processed chunk {i+1}/{len(train_chunks_x)}")
            
            del X_chunk, y_chunk
            gc.collect()
        
        # Process test data
        logger.info("Processing test data...")
        test_chunks_x, test_chunks_y = process_generator(
            test_gen,
            chunk_size=chunk_size,
            n_jobs=n_jobs
        )
        
        # Preprocess test chunks
        processed_test_chunks_x = []
        for X_chunk in test_chunks_x:
            processed_chunk = preprocessor.transform(X_chunk)
            processed_test_chunks_x.append(processed_chunk)
            del X_chunk
            gc.collect()
        
        # Save models and preprocessor
        for name, model in models.items():
            joblib.dump(model, MODEL_DIR / f"{name.lower()}_model.pkl")
        joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")
        
        return models, (processed_test_chunks_x, test_chunks_y)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def predict_chunked(model, X_chunks):
    """Make predictions in chunks"""
    predictions = []
    
    try:
        for i, X_chunk in enumerate(X_chunks):
            chunk_predictions = model.predict(X_chunk)
            predictions.append(chunk_predictions)
            logger.info(f"Processed prediction chunk {i+1}/{len(X_chunks)}")
            
            del X_chunk
            gc.collect()
        
        return np.concatenate(predictions)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.error(f"Chunk shapes: {[x.shape for x in X_chunks]}")
        raise