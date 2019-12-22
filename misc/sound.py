from gtts import gTTS
import cyrtranslit
import io
def main():
    # f = io.open("text.txt", "r", encoding='utf-8')
    # f1 = f.readlines()
    # f2 = io.open("text1.txt", 'w', encoding='utf-8')

    # for line in f1:
        line = 'Боря, будь внимательнее'
        out = cyrtranslit.to_latin(line, 'ru')
        out = "".join(c for c in out if c not in ['!', '.', ':', "'", '?', ' ', '-', '\'', ',', '\n'])
        # f2.writelines(out)
        tts = gTTS(line, lang='ru')
        tts.save(out[:26] + '.mp3')

if __name__ == '__main__':
    main()