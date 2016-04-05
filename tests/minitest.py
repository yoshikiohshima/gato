import sphero
import pyglet

s = sphero.Sphero()

def on_activate():
    print("connect sphero")
    try:
        s.connect()
    except:
        print("err!")
        s.close()

    print( """Bluetooth info:name: %s \nbta: %s """ %
           (s.get_bluetooth_info().name, s.get_bluetooth_info().bta))

if __name__ == '__main__':
    on_activate()
    s.set_rgb(0, 0, 255)

