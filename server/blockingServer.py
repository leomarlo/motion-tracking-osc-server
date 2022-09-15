from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from config import CONFIG
CONFIG.loadMyIP()

client = SimpleUDPClient(CONFIG.clientIP, CONFIG.clientPort) 


def print_handler(address, *args):
    print(f"{address}: {args}")
    client.send_message("/some/address", 123) 


def default_handler(address, *args):
    print(f"DEFAULT {address}: {args}")
    client.send_message("/some/address", "something") 


dispatcher = Dispatcher()
dispatcher.map("/something/*", print_handler)
dispatcher.set_default_handler(default_handler)



server = BlockingOSCUDPServer((CONFIG.myIP, CONFIG.serverPort), dispatcher)
server.serve_forever()  # Blocks forever