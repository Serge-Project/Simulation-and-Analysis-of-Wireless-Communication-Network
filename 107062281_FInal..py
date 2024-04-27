# !pip install panda
import csv
import time
import random
import math
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt


packet_ids = set() # to store generated IDs

class Node:
    def __init__(self, node_id):
        self.id = node_id
        self.buffer = []
        self.power = None
        self.timestamp = None

    def receive_packet(self, packet):
        self.buffer.append(packet)
        self.timestamp = time.time()
    

class SourceNode(Node):
    def __init__(self, node_id, x, y, power, attenuation_factor):
        super().__init__(node_id)
        self.x = x        
        self.y = y 
        self.power = power
        self.attenuation_factor = attenuation_factor

    def generate_packet(self, destination_id, power):
        packet = Packet(self.id, destination_id, power)
        self.buffer.append(packet)
        self.power *= self.attenuation_factor  # Attenuate the power
        return packet
    


class RelayNode(Node):
    def __init__(self, node_id, x, y, power, attenuation_factor):
        super().__init__(node_id)
        self.x = x
        self.y = y
        self.power = power
        self.attenuation_factor = attenuation_factor

    def receive_packet(self, packet, distance):        
        decoded_packet = self.decode_packet(packet, distance)
        if decoded_packet is not None:
            self.attenuate_power()
        return decoded_packet

    def decode_packet(self, packet, distance):
        # calculate path loss and SNR
        SNR_THRESHOLD = 10  # or whatever value you want to use
        
        path_loss, snr = self.calculate_path_loss(packet.power, distance)

        # check if SNR is above a threshold
        if snr < SNR_THRESHOLD:
            print(f"Packet {packet.timestamp:.3f} dropped at Relay {self.id} due to low SNR ({snr:.3f})")
            return None

        # attenuate the signal
        attenuated_power = packet.power - path_loss

        # update the packet
        decoded_packet = packet
        decoded_packet.power = attenuated_power
        decoded_packet.source_to_relay_path_loss = path_loss
        return decoded_packet


    def calculate_path_loss(self, power, distance):
        # define some parameters
        frequency = 2.4e9  # 2.4 GHz
        wavelength = 3e8 / frequency
        tx_gain = 1  # gain of the transmitting antenna
        rx_gain = 1  # gain of the receiving antenna
        PLd0 = 62.3  # path loss at the reference distance
        n = 2.7  # path loss exponent
        noise_floor = -100  # noise floor in dBm
        interference_level = -80  # interference level in dBm

        # calculate path loss
        path_loss = PLd0 + 10 * n * np.log10(distance / wavelength) \
                    + 10 * np.log10(power * tx_gain * rx_gain)
        # calculate received signal power
        received_power = (power - path_loss)
        # calculate noise power
        noise_power = (10 ** ((noise_floor - 30) / 10))
        # calculate interference power
        interference_power = (10 ** ((interference_level - 30) / 10))
        # calculate SNR
        snr = received_power / (noise_power + interference_power)
        return path_loss, snr
    

    def attenuate_power(self):
        self.power *= self.attenuation_factor  # Attenuate the relay node's power

class DestinationNode(Node):
    def __init__(self, node_id, x, y):
        super().__init__(node_id)
        self.x = x
        self.y = y

    def receive_packet(self, packet, distance):
        # calculate path loss and SNR
        SNR_THRESHOLD = 10  # or whatever value you want to use
        
        path_loss, snr = self.calculate_path_loss(packet.power, distance)
        

        # check if SNR is above a threshold
        if snr < SNR_THRESHOLD:
            print(f"Packet {packet.timestamp:.3f} dropped at Destination {self.id} due to low SNR ({snr:.3f})")
            return None

        # attenuate the signal
        attenuated_power = packet.power - path_loss
        
        # update the packet
        decoded_packet = packet
        decoded_packet.power = attenuated_power
        decoded_packet.relay_to_destination_path_loss = path_loss
        return decoded_packet
    
    def calculate_path_loss(self, power, distance):
        # define some parameters
        frequency = 2.4e9  # 2.4 GHz
        wavelength = 3e8 / frequency
        tx_gain = 1  # gain of the transmitting antenna
        rx_gain = 1  # gain of the receiving antenna
        PLd0 = 62.3  # path loss at the reference distance
        n = 2.7  # path loss exponent
        noise_floor = -100  # noise floor in dBm
        interference_level = -80  # interference level in dBm

        # calculate path loss
        path_loss = PLd0 + 10 * n * np.log10(distance / wavelength) \
                    + 10 * np.log10(power * tx_gain * rx_gain)

        # calculate received signal power
        received_power = power - path_loss

        # calculate noise power
        noise_power = 10 ** ((noise_floor - 30) / 10)

        # calculate interference power
        interference_power = 10 ** ((interference_level - 30) / 10)

        # calculate SNR
        snr = received_power / (noise_power + interference_power)

        return path_loss, snr
    
        
class Packet:
    def __init__(self, source_id, destination_id, power):
        self.source_id = source_id
        self.destination_id = destination_id
        self.ID = generate_packet_id()
        self.timestamp = time.time()
        self.power = power
        self.source_to_relay_path_loss = 0
        self.relay_to_destination_path_loss = 0
        self.attempt = 3

def generate_packet_id():
    """
    Generates a random but distinct ID for a packet.
    """
    while True:
        # Generate a random integer between 1 and 1000
        packet_id = random.randint(1000, 5000)
        
        # Check if the ID is already in the set
        if packet_id not in packet_ids:
            # Add the ID to the set
            packet_ids.add(packet_id)
            # Return the ID
            return packet_id

def calculate_channel_quality_source_to_relay(source_node, relay_node):
    # define some parameters
    frequency = 2.4e9  # 2.4 GHz
    wavelength = 3e8 / frequency
    tx_gain = 1  # gain of the transmitting antenna
    rx_gain = 1  # gain of the receiving antenna
    PLd0 = 62.3  # path loss at the reference distance
    n = 2.7  # path loss exponent
    sigma = 4.0  # standard deviation of log-normal shadowing in dB

    # calculate distance between source and relay nodes
    distance = np.sqrt((source_node.x - relay_node.x)**2 + (source_node.y - relay_node.y)**2)

    # calculate path loss with log-normal shadowing
    shadowing = np.random.normal(0, sigma)
    path_loss = PLd0 + 10 * n * np.log10(distance / wavelength) \
                + shadowing \
                + 10 * np.log10(source_node.power * tx_gain * rx_gain)

    # calculate channel quality with log-normal shadowing
    shadowing = np.random.normal(0, sigma)
    channel_quality = 1 / (path_loss + shadowing)

    return channel_quality

def calculate_channel_quality_relay_to_destination(relay_node, dest_node):
    # define some parameters
    frequency = 2.4e9  # 2.4 GHz
    wavelength = 3e8 / frequency
    tx_gain = 1  # gain of the transmitting antenna
    rx_gain = 1  # gain of the receiving antenna
    PLd0 = 62.3  # path loss at the reference distance
    n = 2.7  # path loss exponent
    sigma = 4.0  # standard deviation of log-normal shadowing in dB

    # calculate distance between relay and destination nodes
    distance = np.sqrt((relay_node.x - dest_node.x)**2 + (relay_node.y - dest_node.y)**2)

    # calculate path loss with log-normal shadowing
    shadowing = np.random.normal(0, sigma)
    path_loss = PLd0 + 10 * n * np.log10(distance / wavelength) \
                + shadowing \
                + 10 * np.log10(relay_node.power * tx_gain * rx_gain)

    # calculate channel quality with log-normal shadowing
    shadowing = np.random.normal(0, sigma)
    channel_quality = 1 / (path_loss + shadowing)

    return channel_quality

def Initialize_relay_nodes(source_nodes, relay_nodes, power):
    packets = []
    temp_check = []
    for relay_node in relay_nodes:
        while len(relay_node.buffer) < 2:
            source_node = random.choice(source_nodes)
            packet = source_node.generate_packet(destination_node.id, power)

            distance = np.sqrt((source_nodes[0].x - relay_node.x)**2 + (source_nodes[0].y - relay_node.y)**2)
            decoded_packet = relay_node.receive_packet(packet, distance)
            if decoded_packet is not None:
                packets.insert(0, decoded_packet)
                relay_node.buffer.insert(0, decoded_packet)
                temp_check.append(decoded_packet.ID)
            else:
                continue
                
    return relay_nodes, packets, temp_check

def Source_to_relay_channel_quality(relay_nodes):
    # Initialize list of (relay_node, channel_quality) tuples
    relay_qualities = []

    # Calculate channel quality for each relay node and store in list
    for relay_node in relay_nodes:
        channel_quality = calculate_channel_quality_source_to_relay(source_nodes[0], relay_node)
        relay_qualities.append((relay_node, channel_quality))

    # Sort list by channel quality in descending order
    relay_qualities.sort(key=lambda x: x[1], reverse=True)

    # Extract list of relay nodes in order of decreasing channel quality
    best_relay_nodes = [x[0] for x in relay_qualities]
    return best_relay_nodes


def Transmit_packet_to_relay(best_relay_nodes, source_nodes, packet):
    
    best_relay_node = None
    for relay_node in best_relay_nodes[:3]:
        if len(relay_node.buffer) < 4:
            best_relay_node = relay_node
            break
                 # transmit packet to the best relay node
    if best_relay_node:
        distance = np.sqrt((source_nodes[0].x - best_relay_node.x)**2 + (source_nodes[0].y - best_relay_node.y)**2)
        decoded_packet = best_relay_node.receive_packet(packet, distance)
        if decoded_packet != None:
            best_relay_node.buffer.insert(0, decoded_packet)
            return True
    return False
    

def Relay_to_Destination_channel_quality(relay_nodes, destination_node):

    # Initialize list of (relay_node, channel_quality) tuples
    relay_qualities_to_destination = []

    # Calculate channel quality for each relay node and store in list
    for relay_node in relay_nodes:
        channel_quality_to_destination = calculate_channel_quality_relay_to_destination(relay_node, destination_node)
        relay_qualities_to_destination.append((relay_node, channel_quality_to_destination))

    # Sort list by channel quality in descending order
    relay_qualities_to_destination.sort(key=lambda x: x[1], reverse=True)

    # Extract list of relay nodes in order of decreasing channel quality
    best_relay_nodes_to_destination = [x[0] for x in relay_qualities_to_destination]
    
    return best_relay_nodes_to_destination
def Transmit_packet_to_Destination(best_relay_nodes_to_destination):
    
    best_relay_node_to_destination = None
    
    # Choose the best relay node that also has space in its buffer
    for relay_node in best_relay_nodes_to_destination[:5]:
        if len(relay_node.buffer) > 0:
            best_relay_node_to_destination = relay_node
            break

    # transmit packet to the destination
    if best_relay_node_to_destination:
        packet = best_relay_node_to_destination.buffer.pop()
        distance = np.sqrt((best_relay_node_to_destination.x - destination_node.x)**2 + (best_relay_node_to_destination.y - destination_node.y)**2)
        Final_Packet = destination_node.receive_packet(packet, distance)
        
        if Final_Packet != None:
            destination_node.buffer.append(Final_Packet)
            return True
        
    return False
def phi_func(theta,beta):
    y = theta-(1/(2*beta))
    return y 
def psi_func(theta,beta):
    y = theta+(1/(2*beta))
    return y
def theta_func(R):
    y = (2**R) - 1
    return y
def belta_func(m,R):
    y = (1/2*math.pi)*math.sqrt(m/(2**((2*R)-1)))
    return y
def rayleigh_fading_channel_power(power):
    h  = 1/np.sqrt(2)*(np.random.randn()+1j*np.random.randn())
    h_abs = np.abs(h)
    y = power*h_abs
    return y
def transmit(source_nodes, relay_nodes, destination_node, power):
    
# =======================================================================================================
#             Initialize Every relay buufer with half their capacity with packets
# =======================================================================================================

#     relay_nodes, packets, temp_check = Initialize_relay_nodes(source_nodes, relay_nodes, power)
    
# =============================================================================
# Parameters
# =============================================================================


    h_sr = -90 # Average channel power of S-R channel (dBm); pathloss implicitly included
    h_rd = -90 # Average channel power of R-D channel (dBm); pathloss implicitly included
    P_s = 15 # Source transmit power (dBm)
    P_r = 15 # Relay transmit power (dBm)
    N_r = -95 # Noise power at relay (dBm)
    N_d = -95 # Noise power at destination (dBm)
    
    
    N = 200 # Number of data bits per packet
    m = 220 # Packet length after encoding
    
# =============================================================================
# Block error rate setting
# =============================================================================

    gamma_sr_db = P_s+h_sr-N_r # SNR source to relay
    gamma_sr = 10**(gamma_sr_db/10)
    gamma_sr_before_channel = gamma_sr
    gamma_rd_db = P_r+h_rd-N_d #SNR relay to desptination
    gamma_rd = 10**(gamma_rd_db/10)
    gamma_rd_before_channel = gamma_rd
    
# =======================================================================================================
#                                     Variable Initialization
# =======================================================================================================

    t = 0 # Time index
    AOI = 0
    simulation_time = 5
    packets = []

# =============================================================================
# The following is used for computing the theoretical AoI. 
# =============================================================================


    m_H = m/2 # Packet length in half-duplex radio
    R_H = 2*N/m # Coding rate in half-duplex radio
    belta_H = belta_func(m_H,R_H)
    theta_H = theta_func(R_H)
    phi_H = phi_func(theta_H,belta_H)
    psi_H = psi_func(theta_H,belta_H)
    
    
# =======================================================================================================
#                                      Generate new packets
# =======================================================================================================

    simulated_time = []
    Number_of_packets_sent = []
    Number_of_packets_received = []
    aoi_value = []
    aoi_average = []
    

    simulated_time.append(0)
    Number_of_packets_sent.append(0)
    Number_of_packets_received.append(0)
    aoi_value.append(0)
    aoi_average.append(0)

    time_slot = simulation_time/5
    idx_simulated_time = time_slot
    # simulated_time.append(idx_simulated_time)

    
    for p in range(0, 1500):
        source_node = random.choice(source_nodes)
        packet = source_node.generate_packet(destination_node.id, power)
        packets.insert(0, packet)
       
    
# ========================================================================================================================================
#                                         Actual Simulation of packets transmission
# ========================================================================================================================================
    
    count = 0
    while t < simulation_time:

        if t >= idx_simulated_time:
            simulated_time.append(idx_simulated_time)
            Number_of_packets_sent.append(count)
            Number_of_packets_received.append(len(destination_node.buffer))
            aoi_value.append(AOI)
            aoi_average.append(AOI/t)
            idx_simulated_time += time_slot


        current_sim_time = time.time()
        #create new packets in case all packets has been sent before simulation time
        if len(packets) == 0:
            for p in range(0, 1000):
                source_node = random.choice(source_nodes)
                packet = source_node.generate_packet(destination_node.id, power)
                packets.insert(0, packet) 
        # break if simulation time is over
        if t >= simulation_time:
            break
            
        x_sr = random.uniform(0,1) # Uniform random variable in (0,1); 
                               # used for deciding the transmission outcome of the S-R channel
        x_rd = random.uniform(0,1) # Uniform random variable in (0,1);  # used for deciding the transmission outcome of the R-D channel     
        gamma_sr = rayleigh_fading_channel_power(gamma_sr_before_channel)
        gamma_rd = rayleigh_fading_channel_power(gamma_rd_before_channel)
        
        E_H_SR = 1 - (belta_H*gamma_sr*(math.exp(-(phi_H/gamma_sr))-math.exp(-(psi_H/gamma_sr)))) # Packet error rate for S-R channel
        E_H_RD = 1 - (belta_H*gamma_rd*(math.exp(-(phi_H/gamma_rd))-math.exp(-(psi_H/gamma_rd)))) # Packet error rate for R-D channel

        packet = packets.pop()
        packet.timestamp = time.time()

        if x_sr > E_H_SR: # A packet is transmitted successfully from S to R
            # The Source to Relay channel quality of every relay nodes
            best_relay_nodes = Source_to_relay_channel_quality(relay_nodes)
            count += 1

            #Transfer Urgent packets to the best and available relay node
            report = Transmit_packet_to_relay(best_relay_nodes, source_nodes, packet)
            
            if report == False:
                if packet.attempt > 0:
                    packet.attempt -= 1
                    packets.insert(0, packet)
                else:
                    print("Packet ", packet.ID, " has been dropped at Relay")
                    
        else:
            if packet.attempt > 0:
                packet.attempt -= 1
                packets.insert(0, packet)
            else:
                print("Packet ", packet.ID, " has been dropped at Relay")
        
                       
# =======================================================================================================
#                                  Sending Packets to Destination
# =======================================================================================================
        if x_rd > E_H_RD:
         
            # The Relay to Destination channel quality of every relay nodes
            best_relay_nodes_to_destination = Relay_to_Destination_channel_quality(relay_nodes, destination_node)


            # Choose the best relay node, pop the end, send it to the destination
            Dest_status = Transmit_packet_to_Destination(best_relay_nodes_to_destination)
    
# =======================================================================================================
#                     After Final_Packet arrive at D we calculate the AoI and update t
# =======================================================================================================
            current = 0  
            if Dest_status == True:
                received_packet = destination_node.buffer[len(destination_node.buffer) - 1]
                
                current = time.time() - received_packet.timestamp
                temp = time.time() - current_sim_time
                t += current
                if t >= simulation_time:
                    break
                AOI += current
                continue
        
        
        
    average_AOI = AOI/simulation_time
    
    print()
    print("Simulation Time ", simulation_time)
    print("Final AOI ", AOI)
    print("average_AOI",average_AOI)
    print("Number of packets sent ", count)
    print("Number of Packets received ", len(destination_node.buffer))
    print("Number of packet unsent ", len(packets))
    print()
    print()
    
    
#     print("simulated_time ", simulated_time)
#     print("Number_of_packets_sent ", Number_of_packets_sent)
#     print("Number_of_packets_received ", Number_of_packets_received)
#     print("aoi_value ", aoi_value)
#     print("aoi_average ", aoi_average)
#     print()

    r = []
    for relay in relay_nodes:
        r.append(len(relay.buffer))
    print("Number of packets dropped at relay after simulation time")
    print(r)
    
    print()
    print()

    plt.plot(simulated_time, Number_of_packets_sent, marker='o', label='Packets Transferred')
    plt.plot(simulated_time, Number_of_packets_received, marker='o', label='Packets Received')
    
#     fok u fe 2 plot diferan youn pou packet yo you pou AOI yo 
    
#     plt.plot(simulated_time, aoi_value, marker='o', label='Age of Information')
#     plt.plot(simulated_time, aoi_average, marker='o', label='Average age of Information')

    plt.xlabel('Simulation Time')
    plt.ylabel('Value')
    plt.title('Simulation Time vs. Value')
    plt.legend()
    plt.grid(True)

    # Display the graph
    plt.show()
    
    plt.plot(simulated_time, aoi_value, marker='o', label='Age of Information')
    plt.plot(simulated_time, aoi_average, marker='o', label='Average age of Information')
    
    plt.xlabel('Simulation Time')
    plt.ylabel('Value')
    plt.title('Simulation Time vs. Value')
    plt.legend()
    plt.grid(True)

    # Display the graph
    plt.show()
    
    

# =============================================================================
# Create Source nodes, relay nodes, 
# =============================================================================


source_node1 = SourceNode('s_1', -4, 8, 100, 0.9)
source_node2 = SourceNode('s_2', -2, 5, 100, 0.9)
source_node3 = SourceNode('s_3', 3, 3, 100, 0.9)
source_node4 = SourceNode('s_4', -1, -9, 100, 0.9)
source_node5 = SourceNode('s_5', 2, -5, 100, 0.9)


relay_node1 = RelayNode('r_1', -5, 0, 100, 0.9)
relay_node2 = RelayNode('r_2', -4, 0, 100, 0.9)
relay_node3 = RelayNode('r_3', -3, 0, 100, 0.9)
relay_node4 = RelayNode('r_4', -2, 0, 100, 0.9)
relay_node5 = RelayNode('r_5', -1, 0, 100, 0.9)
relay_node6 = RelayNode('r_6', 0, 0, 100, 0.9)
relay_node7 = RelayNode('r_7', 1, 0, 100, 0.9)
relay_node8 = RelayNode('r_8', 2, 0, 100, 0.9)
relay_node9 = RelayNode('r_9', 4, 0, 100, 0.9)
relay_node10 = RelayNode('r_10', 5, 0, 100, 0.9)

destination_node = DestinationNode(100, 0, -10)

relay_nodes = [relay_node1, relay_node2, relay_node3, relay_node4, relay_node5, relay_node6, relay_node7, relay_node8, relay_node9, relay_node10]
source_nodes = [source_node1, source_node2, source_node3, source_node4, source_node5]

destination_node = destination_node

# simulate transmission
transmit(source_nodes, relay_nodes, destination_node, 10000)





