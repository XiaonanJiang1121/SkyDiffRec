import unittest

from vlm_skyfind.adapters import registry


class RegistryTest(unittest.TestCase):
    def test_geochat_uses_its_own_runtime(self):
        module_name, class_name = registry._ADAPTERS["geochat-7b"]
        self.assertEqual(module_name, "vlm_skyfind.adapters.geochat")
        self.assertEqual(class_name, "GeoChatAdapter")


if __name__ == "__main__":
    unittest.main()
