from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from config import CONFIG
CONFIG.loadMyIP()
## ONLY FOR TESTING. DONT CALL THIS FUNCTION OTHERWISE
# CONFIG.onlyReceive()
# CONFIG.localTesting()

client = SimpleUDPClient(CONFIG.clientIP, CONFIG.clientPort) 


def print_handler(address, *args):
    print(f"{address}: {args}")
    client.send_message("/some/address", 123) 


def default_handler(address, *args):
    print(f"DEFAULT {address}: {args}")
    if CONFIG.ONLY_RECEIVE:
        return
    if args[0] == 'tilts':
        client.send_message(address, (args[1], args[2])) 


dispatcher = Dispatcher()
dispatcher.map("/something/*", print_handler)
dispatcher.set_default_handler(default_handler)



server = BlockingOSCUDPServer((CONFIG.myIP, CONFIG.serverPort), dispatcher)
server.serve_forever()  # Blocks forever