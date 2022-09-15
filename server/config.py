from socket import socket, AF_INET, SOCK_DGRAM, gethostname
# print(socket.gethostbyname(socket.gethostname()))

class CONFIG:
    clientIP = "192.168.0.46"
    localhost = "127.0.0.1"
    myIP = "192.168.0.45hallo" # "127.0.0.1"
    myHostName = ""
    serverPort = 9999
    clientPort = 53535

    # def __init__(self) -> None:
    #     pass

    def loadMyIP():
        s = socket(AF_INET, SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        hostName = gethostname()
        CONFIG.myHostName = hostName
        CONFIG.myIP = s.getsockname()[0]

# config = CONFIG()
# config.init()
# CONFIG.init()
print(CONFIG.myHostName, CONFIG.myIP)

# from netifaces import interfaces, ifaddresses, AF_INET
# for ifaceName in interfaces():
#     print(ifaddresses(ifaceName))
#     # addresses = [i['addr'] for i in ifaddresses(ifaceName).setdefault(AF_INET, [{'addr':'No IP addr'}] )]
#     # print(' '.join(addresses))