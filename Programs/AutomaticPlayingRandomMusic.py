import random, os

music_dir = 'D:\\Audios\\Songs'
songs = os.listdir(music_dir)

song = random.randint(0, len(songs))

# Prints the song name
print(songs[song])

os.startfile(os.path.join(music_dir, songs[song]))
