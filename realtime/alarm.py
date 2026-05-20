import winsound

def play_alarm():

    duration = 500
    freq = 1000

    winsound.Beep(freq, duration)