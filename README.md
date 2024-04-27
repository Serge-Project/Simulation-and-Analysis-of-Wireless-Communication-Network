Spring 2023 – Cooperative Communications and Networking
    Simulation and Analysis of Wireless Communication Network:
A Case Study
    Serge Rigaud Joseph, 107062281
June 14, 2023
Link to Project Demo:
    https://drive.google.com/drive/folders/1AjFotDAq-EVGkhGnvtrpd_Qico2isrDj?usp=share_link

Overview

The goal of this project is to better the reliability and performance of wireless communication networks. The problem under consideration is signal quality degradation and packet loss. The signal problem is mainly caused by path loss, interference and noise. In this study we aim to find ways to minimize the impacts these factors can have on the communication quality, as well as to increase the number of packets transferred over time utilizing packet retransmission.
To attend this objective, we use the help of relay nodes to strengthen signal strength and source to destination signal quality. Their job is to receive the packets decode it and then forward it to the destination node. We also use the help of buffers and packet retransmission to help minimize the packet loss problem.

1. Introduction

Our daily life is based upon transferring information from one point to another, whether it’s using email, cell phone, or other communication method. However, so many times our call do not go through, or messages are not delivered, those problems are due to the variety of difficulties that might reduce signal quality and impact overall performance. Signal quality can degrade due to variables such as route loss, interference, and noise. As a result, there is an increasing need to explore and develop methods for mitigating these impacts and improving the reliability and performance of wireless communication networks.

The goal of this research is to alleviate the problem of signal deterioration by implementing relay nodes. Relay nodes act as middlemen between the source and destination nodes, allowing data packet delivery. We can improve the signal strength and quality between communicating nodes by strategically installing relay nodes and utilizing relevant algorithms.

The primary goal of this research is to investigate how effective relay nodes are at mitigating the impacts of route loss, interference, and noise in wireless communication systems. We investigate how channel quality measurements might be used to rank and pick the best relay nodes for packet transmission. We also use path loss models, antenna gains, noise levels, and fading channel models to determine channel power and optimize the relay selection process.

We hope to improve the dependability and performance of wireless communication systems by employing these ideas, which will result in better signal quality and higher transmission success rates. The findings of this study may help to enhance wireless communication technology, which will benefit a variety of applications such as mobile networks, the Internet of Things (IoT), and smart cities.



1 System Model

The code defines classes for nodes in a wireless communication network, including as source nodes, relay nodes, and destination nodes. Each node contains features such as an ID, position, power, and a buffer to store packets. The code also includes functions for generating packet IDs, calculating channel quality, and transmitting packets between nodes.

The parameters and settings for the wireless communication system are given in the main function "transmit". The function fills the buffers of relay nodes with packets from source nodes. It then transmits data from source nodes to relay nodes and from relay nodes to the destination node. The channel quality is determined, and packets are sent using the best available relay nodes with enough buffer capacity. The code contains options for adjusting the block error rate and signal-to-noise ratio.

Overall, the code simulates packet transmission in a wireless communication network with many source nodes, relay nodes, and one destination node while accounting for path loss, attenuation, channel quality, and buffer management. The decode and forward method is used to send packets.


2 Define Section Title
To implement the system model described earlier, we can follow a step-by-step approach. Here's a detailed description of the method:


1- Define Node Classes:
Begin by designing classes for the wireless communication network's nodes. This covers the source and relay nodes as well as the destination node. Each node class should include features such as an ID, position, power level, and a buffer for packet storage. Implement techniques for calculating channel quality, transmitting packets, and managing the buffer as well.

2- Initialize System Parameters:   
Configure the wireless communication system's characteristics and settings. Defining the number of source and relay nodes, transmission power levels, block error rate, and signal- to-noise ratio are all examples of this. These settings will be applied throughout the simulation.

3- Generate Packet IDs:
Create a function that generates unique packet IDs for each packet sent via the system. This can be accomplished by using a counter or a random number generator.

4- Source Node Initialization:
Create instances of the source node class to initialize the source nodes. Set their
placements, transmission power levels, and buffer sizes.

5- Relay Node Initialization:
Create instances of the relay node class to represent the network's relay nodes. Set their placements, transmission power levels, and empty buffers to default.

6- Buffer Initialization:
Fill the relay node buffers with packets from the source nodes to get them started. Assign packets to relay nodes based on their proximity or any other criteria that you wish.

7- Transmission Process:
Start the transmission procedure. Begin by sending packets from source nodes to relay nodes. Using relevant models (e.g., path loss, attenuation), calculate the channel quality between each source node and relay node. Choose the optimal relay node(s) with enough buffer space to broadcast the packets based on the channel quality.

8- Relay-to-Destination Transmission
Continue sending packets from the relay nodes to the target node. Determine the quality of the channel between each relay node and the destination node. Select the relay node(s) with the best channel quality and available buffer capacity to transmit the packets, as in the previous phase.

9- Packet Reception and Buffer Management:
At the relay and destination nodes, implement methods for packet reception and buffer management. According to the set block error rate, update the buffer state, delete successfully received packets, and discard or retransmit packets with errors.

10- Simulation Termination:
Choose a simulation termination condition. This could be based on a given amount of transmitted packets, a preset simulation period, or any other criterion relevant to the unique system needs.

You may effectively implement the system model and simulate packet transmission in a wireless communication network by using this method. Remember to test and validate the implementation to ensure that it matches the expected functionality and behavior.
COM, NTHU 3




3 Results and Discussions

Simulation/Experiment Configurations:

Network Topology:
Define the wireless communication system's network topology, including the number and placement of source nodes, relay nodes, and the destination node. This may include node placements or specialized connectivity patterns.

Transmit Power Levels:
Configure the transmission power settings for the source and relay nodes. This parameter affects the overall system performance by determining the strength of the sent signal.

Channel Model:
Select an appropriate channel model to replicate the wireless propagation characteristics. This could be based on path loss models, fading models (e.g., Rayleigh, Rician), or other wireless- specific channel models.

Block Error Rate (BLER):
Set the system's tolerable block error rate. The BLER represents the proportion of packets that can contain errors before being declared lost or requiring retransmission.

Simulation Time:
Set the duration of the simulation or experiment. This parameter specifies the total amount of time that the system behavior will be watched and assessed.

Parameters Subject to Change:
Number of Nodes: Vary the number of source nodes and relay nodes in the network to observe the impact on system performance. This can help evaluate scalability and efficiency.

Node Placement:
Change the positions of the nodes in the network to assess the effect of different spatial distributions on the overall system behavior. This can include random placements, clustered deployments, or specific scenarios like line-of-sight or non-line-of-sight conditions.

Transmit Power Levels:
Experiment with different transmission power levels for the source nodes and relay nodes. This can help analyze the trade-off between power consumption and signal quality.

Performance Metrics of Interest:
Packet Delivery Ratio (PDR): Calculate the ratio of successfully delivered packets to the total number of transmitted packets. This metric measures the system's ability to reliably transmit packets from source nodes to the destination node.

End-to-End Delay:
Measure the time taken for a packet to travel from a source node to the destination node. This metric assesses the latency of the system and can be crucial for delay-sensitive applications.

Energy Efficiency:
Evaluate the energy consumption of the system, considering factors such as transmit power levels and node activity. This metric helps assess the system's energy efficiency and sustainability.

You can systematically examine and analyze the behavior and performance of the wireless communication system under various scenarios by splitting the simulation/experiment setups into subsections depending on the parameters subject to change or the performance metrics of interest.



4 Summary
To summarize, in order to create the system, we must first define the simulation/experiment setups, which include the network architecture, transmit power levels, channel model, block error rate, and simulation time. These setups serve as the basis for system simulation.

Following that, we identify the variables that can be changed, such as the number of nodes, node positioning, and transmit power levels. We may observe the influence of these parameters on system performance and analyze trade-offs by adjusting them.

Finally, we discuss key performance indicators such as packet delivery ratio (PDR), end-to-end delay, throughput, and energy efficiency. We can use these measures to evaluate the system's dependability, latency, efficiency, and energy usage.

We may systematically investigate numerous scenarios and evaluate the behavior and performance of the wireless communication system by structuring the implementation details in this manner.

