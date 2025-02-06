from transformers import PreTrainedTokenizer

from sccello.src.utils import data_loading

class scCelloPreCollator(PreTrainedTokenizer):
    def __init__(self, *args, **kwargs) -> None:

        self.token_dictionary = data_loading.get_prestored_data("token_dictionary_pkl")
        
        super().__init__(mask_token="<mask>", pad_token="<pad>", cls_token="<cls>")
        
        assert self.pad_token_id == 0
        assert self.mask_token_id == 1

        self.padding_side = "right"
        self.model_input_names = kwargs.get("model_input_names")
    
    def convert_ids_to_tokens(self, value):
        return self.token_dictionary.get(value)

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        return self.token_dictionary.get(token)
    
    def __len__(self):
        return len(self.token_dictionary)

    def get_vocab(self):
        return self.token_dictionary