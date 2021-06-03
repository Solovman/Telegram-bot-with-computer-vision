from imageai.Classification.Custom import ClassificationModelTrainer

model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsDenseNet121()
model_trainer.setDataDirectory('dataset')
model_trainer.trainModel(num_objects=4, num_experiments=10, enhance_data=True, batch_size=32, show_network_summary=True)
