
from pythonosc.udp_client import SimpleUDPClient

class Client:
    client : SimpleUDPClient

    address = {
        "energy": "/dm/energy",
        "centervelocity": "/dm/centervel",
        "sizedifference": "/dm/sizediff",
        "gyroenergy": "/dm/gyroenergy",
        "gyroEWMAEnergyDerivative": "/dm/gyroewmaderiv",
        "gyroEnergyDerivative": "/dm/gyroewmaderiv",
        "gyroEWMAEnergy":"/dm/gyroewma",
        "gyroEWMAOfEWMADerivative": "/dm/gyroewmaderivewma"}
    
    def setClient(ip, port):
        Client.client = SimpleUDPClient(ip, port) 