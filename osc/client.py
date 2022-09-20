
from pythonosc.udp_client import SimpleUDPClient

class Client:
    client : SimpleUDPClient
    
    def setClient(ip, port):
        Client.client = SimpleUDPClient(ip, port) 