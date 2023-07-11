import torch
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import telebot
import requests
from model import Generator


def init_model():
    network = Generator(5)
    network.load_state_dict(
        torch.load('weights/monet_to_photo/gen_ba_200epochs.pt'))
    return network


token = 'TOKEN'
bot = telebot.TeleBot(token)
network = init_model()
to_tensor = transforms.Compose(
    [transforms.PILToTensor(), transforms.Resize((1024, 1024), Image.BICUBIC)])


@bot.message_handler(content_types=['photo'])
def get_message(message):
    # saving the photo
    size = message.photo[len(message.photo)-1]
    file = bot.get_file(size.file_id)
    URL = f'https://api.telegram.org/file/bot{token}/{file.file_path}'
    response = requests.get(URL)
    open(f'{file.file_id}.png', 'wb').write(response.content)

    # passing the photo through model
    img = Image.open(f'{file.file_id}.png')
    img = to_tensor(img)
    img = network(img.float())

    # sending the image back
    resize_back = transforms.Compose(
        [transforms.Resize((size.height, size.width), Image.BICUBIC)])
    save_image(resize_back(img), f'{file.file_id}.png')
    result = open(f'{file.file_id}.png', 'rb')
    bot.send_photo(chat_id=message.chat.id, photo=result)

    # celaring the files
    os.remove(f'{file.file_id}.png')


bot.polling(none_stop=True, interval=0)
