from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config

RESPONSE = {
    'Amount':None,
    'Merchant': None,
	'Date': None,
	'Address': None
}

KEYS = ['Amount', 'Merchant', 'Date', 'Address']
PREFIX_BIO = ['B', 'I']

TEXT_STR = '521 Hoang Van\n Hoang Van Thu Q.Tan Binh\n DON\n 23/06/2018(Thu\n 08:10\n Temakionigiri\n 16.000\n 16,000*\n tinh:\n cong:\n 16,000\n iue\n So don:2-69318 Thu ngan:120013\n khach nhu cau xuat hca don do,\n vien toi\n '

TEXT_STR_1 = '521 Hoang Van Hoang Van Thu Q.Tan Binh'

TEXT_STR_2 = 'CIRCLE K VIETNAM\n 124 Pho Quang. W.9. Phu Nhuan.D. HCM\n Receipt No: 088 02 20180628 0314\n Date: jun, 29 2018 06:12 PM\n Jun 28,\n Cashier: 5473-Diem Tan Thi Le\n Description:\n 1 NESTEA Tea LemON Small 120z*1CP 7,000\n 1 Item(s) (VAT included) 7,000\n CASH\n 7,000\n CHANGE\n '

TEXT_STR_3 = 'CIRCLE K VIETNAM\n\
968 3/2 Street. Ward 15 Dist 11 HCMC\n\
Receipt No: 153 02 20180630 0046\n\
Date: Jun 30, 2018 07:37 AM\n\
Cashier: 5022-Vi Ngo Nguyen Yen\n\
Description:\n\
CK Mixed Drinkime Soda\n\
Noodle with Fried Egg 14,000\n\
DC//C+ VTM C 350 6,000\n\
2 Item(s) (VAT included) 20,000\n\
20,000\n\
CASH\n\
CHANGE\n\
'

def get_data_from_key(data, key):
    output = []

    # B to I
    prefix_key = 'B-' + key
    for idx, pred in enumerate(data['output']):
        if pred == prefix_key:
            output.append(data['input'][idx])
            found = False
            prefix_key_i = 'I-' + key
            for i, pred_i in enumerate(data['output'][idx:]):
                if pred_i == prefix_key_i:
                    found = True
                    output.append(data['input'][i + idx])
                    print(data['input'][i + idx], i, idx)
                elif found == True:
                    break
            break

    if len(output) > 0:
        return ' '.join(output)

    # I only
    prefix_key = 'I-' + key
    found = False
    for idx, pred in enumerate(data['output']):
        if pred == prefix_key:
            found = True
            output.append(data['input'][idx])
        elif found == True:
            return ' '.join(output)




    for prefix in PREFIX_BIO:
        prefix_key = prefix + '-' + key
        print(prefix_key)
        for idx, pred in enumerate(data['output']):
            if pred == prefix_key:
                print(idx)
                output.append(data['input'][idx])

    return output

def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned
        
        print(seq)

    return data_aligned



def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break

        preds = model.predict(words_raw)
        # to_print = align_data({"input": words_raw, "output": preds})
        model.logger.info(preds)
        print(preds)
        # for key in KEYS:
        #     print(key)
        #     value = get_data_from_key({'input': words_raw, 'output': preds}, key)
        #     print(value)

        # for key, seq in to_print.items():
            # model.logger.info(seq)
def load_model():
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    return model

def predict_from_text(model, text):
    lines = text.split('\n')
    words = []
    prediction = []
    for line in lines:
        token = line.strip().split(' ')
        preds = model.predict(token)
        words = words + token
        prediction = prediction + preds

    # print(words)
    # print(prediction)
    response = {}
    for key in KEYS:
        print(key)
        value = get_data_from_key({'input': words, 'output': prediction}, key)
        if len(value) > 0:
            response[key] = value
    return response


def main():
    model = load_model()
    # create instance of config
    # config = Config()

    # # build model
    # model = NERModel(config)
    # model.build()
    # model.restore_session(config.dir_model)

    # create dataset
    # test  = CoNLLDataset(config.filename_test, config.processing_word,
    #                      config.processing_tag, config.max_iter)

    # evaluate and interact
    # DkS, stop evaluate
    # model.evaluate(test)
    # interactive_shell(model)
    test_string = TEXT_STR_3
    response = predict_from_text(model, test_string)

    print(response)

if __name__ == "__main__":
    main()
