from socket import socket, AF_INET, SOCK_DGRAM, gethostname
# print(socket.gethostbyname(socket.gethostname()))

class CONFIG:
    clientIP = "192.168.0.46"
    localhost = "127.0.0.1"
    myIP = "192.168.0.45" # "127.0.0.1"
    myHostName = ""
    serverPort = 53534
    clientPort = 53534
    ONLY_RECEIVE = False
    STOP_SERVER = False
    RECORD_MOBILE_DATA = True
    STORE_RECORDINGS_IN_DB = True
    verbosity = 2

    # def __init__(self) -> None:
    #     pass

    def loadMyIP():
        s = socket(AF_INET, SOCK_DGRAM)
        s.connect(("10.0.0.1", 8080))
        hostName = gethostname()
        CONFIG.myHostName = hostName
        CONFIG.myIP = s.getsockname()[0]
        print('my ip is: ', CONFIG.myIP)
        # CONFIG.myIP = '192.168.2.56' 

    def onlyReceive():
        CONFIG.ONLY_RECEIVE = True

    def localTesting():
        CONFIG.myIP = CONFIG.localhost

    def updteClientIP(clientIP):
        CONFIG.clientIP = clientIP

    def stopServer():
        CONFIG.STOP_SERVER = True
    
    def startRecordingMobile():
        CONFIG.RECORD_MOBILE_DATA = True

    def stopRecordingMobile():
        CONFIG.RECORD_MOBILE_DATA = True
    
    
# config = CONFIG()
# config.init()
# CONFIG.init()

# from netifaces import interfaces, ifaddresses, AF_INET
# for ifaceName in interfaces():
#     print(ifaddresses(ifaceName))
#     # addresses = [i['addr'] for i in ifaddresses(ifaceName).setdefault(AF_INET, [{'addr':'No IP addr'}] )]
#     # print(' '.join(addresses))