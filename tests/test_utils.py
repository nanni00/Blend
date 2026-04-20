import logging
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from blend.utils import init_logger


class InitLoggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger(f"blend_logger_{os.getpid()}")
        self._reset_logger()

    def tearDown(self) -> None:
        self._reset_logger()

    def _reset_logger(self) -> None:
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()

    def test_init_logger_adds_stdout_and_file_handlers(self) -> None:
        with TemporaryDirectory() as tmpdir:
            logfile = Path(tmpdir) / "blend.log"

            logger = init_logger(logfile=logfile, stdout=True)

            file_handlers = [
                handler
                for handler in logger.handlers
                if isinstance(handler, logging.FileHandler)
            ]
            stdout_handlers = [
                handler
                for handler in logger.handlers
                if isinstance(handler, logging.StreamHandler)
                and not isinstance(handler, logging.FileHandler)
                and handler.stream is sys.stdout
            ]

            self.assertEqual(len(file_handlers), 1)
            self.assertEqual(len(stdout_handlers), 1)

    def test_init_logger_does_not_duplicate_handlers(self) -> None:
        with TemporaryDirectory() as tmpdir:
            logfile = Path(tmpdir) / "blend.log"

            init_logger(logfile=logfile, stdout=True)
            logger = init_logger(logfile=logfile, stdout=True)

            file_handlers = [
                handler
                for handler in logger.handlers
                if isinstance(handler, logging.FileHandler)
            ]
            stdout_handlers = [
                handler
                for handler in logger.handlers
                if isinstance(handler, logging.StreamHandler)
                and not isinstance(handler, logging.FileHandler)
                and handler.stream is sys.stdout
            ]

            self.assertEqual(len(file_handlers), 1)
            self.assertEqual(len(stdout_handlers), 1)


if __name__ == "__main__":
    unittest.main()
