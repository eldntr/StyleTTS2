# IPA Phonemizer: https://github.com/bootphon/phonemizer

_special = ['<pad>', '<sos>', '<eos>', '<unk>', '<mask>']
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = _special + list(_punctuation) + list(_letters) + list(_letters_ipa)

# Filter unique symbols
symbols = list(dict.fromkeys(symbols))

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i


class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
        print("Final Vocab Size:", len(dicts))

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                indexes.append(self.word_index_dictionary['<unk>'])
        return indexes


if __name__ == "__main__":
    print(TextCleaner())