from Config import *
from telegram import ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from imageai.Classification.Custom import CustomImageClassification
import os

updater = Updater(token)


# command "/start"
def start_command(update, _: CallbackContext):
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr"Привет, {user.mention_markdown_v2()}\!" + "\n Я бот, созданный различать марки автомобилей по фото",
        reply_markup=ForceReply(selective=False)
    )


# command "/help"
def help_command(update, _: CallbackContext):
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr"{user.mention_markdown_v2()}\!" + " чем я могу тебе помочь?",
        reply_markup=ForceReply(selective=False)
    )


# team recognition of the car brand by photo
def photo_identification(update, _: CallbackContext):
    image = update.message.photo[0].get_file()
    image.download('user_photo.jpg')

    execution_path = os.getcwd()

    prediction = CustomImageClassification()
    prediction.setModelTypeAsDenseNet121()
    prediction.setModelPath(os.path.join(execution_path, 'model_ex-006_acc-0.580153.h5'))
    prediction.setJsonPath(os.path.join(execution_path, 'model_class.json'))
    prediction.loadModel(num_objects=4)

    prediction, probabilities = prediction.classifyImage('user_photo.jpg', result_count=5)

    for eachPrediction, eachProbability in zip(prediction, probabilities):
        update.message.reply_text(str(eachPrediction) + ":" + str(eachProbability))


def main():
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start_command))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(MessageHandler(Filters.photo, photo_identification))
    updater.start_polling()
    updater.idle()


main()