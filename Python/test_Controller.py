from unittest import TestCase
import Controller

class TestController(TestCase):
    def test_pipeline(self):
        ctr = Controller.Controller()
        ctr.pipeline()
