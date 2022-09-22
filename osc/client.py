
from pythonosc.udp_client import SimpleUDPClient

class Client:
    client : SimpleUDPClient

    address = {
        "energy": "/dm/energy",
        "centervelocity": "/dm/centervel",
        "sizedifference": "/dm/sizediff"}
    
    def setClient(ip, port):
        Client.client = SimpleUDPClient(ip, port) 