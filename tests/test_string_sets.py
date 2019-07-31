from libs.utils import string_list_complement

class Test_string_list_complement():
    def setup_string_list_complement(self):
        self.list1 = [
            "banana",
            "dog",
            "person",
            "bonoro",
            "lula"
        ]

        self.list2 = [
            "banana",
            "bonoro",
            "person",
            "MD"
        ]

    def test_string_list_complement(self):
        self.setup_string_list_complement()

        self.list1_2 = string_list_complement(self.list1, self.list2)
        assert len(self.list1_2) == 2

        self.list2_1 = string_list_complement(self.list2, self.list1)
        assert len(self.list2_1) == 1
