class CustomModelDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        base_path: Path,
        # hyperparams: ModelHyperParams,
        logger: logging.Logger,
        shuffle: bool = True,
    ) -> None:


        self._batch_size = hyperparams.training.batch_size
        self._paths = paths
        self._shuffle = shuffle
        self._logger = logger
        self._base_path = base_path
        self.on_epoch_end()

        self._data = Data(self._logger)
        self._data.get_files_links(base_path=Path('./dataset/train'))



    def on_epoch_end(self):
        """
        Shuffles the paths to the images at the end of each epoch if shuffling is enabled.
        """

        if self._shuffle:
            np.random.shuffle(self._paths)

    def __len__(self):
        """
        Returns the number of batches per epoch.

        ### Returns:
        - int: The number of batches in the dataset for each epoch.
        """

        return len(self._paths) // self._batch_size

    def __getitem__(self, index):
        """
        Generates one batch of data.

        ### Args:
        - index (int): Index of the batch to generate.

        ### Returns:
        - tuple: A tuple containing the generated data batch and corresponding labels.

        """

        self._logger.debug("Uploading images")

        return self.data.load_files(index,  self._batch_size)
