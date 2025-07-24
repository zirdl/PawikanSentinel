import unittest
from src.alert_manager.alert_generator import AlertGenerator
from src.alert_manager.alert_delivery import AlertDelivery
from src.notification_service.mock_service import MockNotificationService

class TestAlertManager(unittest.TestCase):

    def test_alert_generator(self):
        generator = AlertGenerator(1)
        self.assertEqual(generator.generate_alert({0: (10, 10)}), "A sea turtle has been detected.")
        self.assertEqual(generator.generate_alert({0: (10, 10)}), "")

    def test_alert_delivery(self):
        service = MockNotificationService()
        delivery = AlertDelivery(service)
        delivery.enqueue_alert("Test Alert")
        delivery.process_queue()
        # This test is basic and mainly checks for crashes.
        # A more advanced test could capture stdout to verify the print statements.

if __name__ == '__main__':
    unittest.main()
