package telemetry

import (
	"log"
	"net"
	"time"
)

// StartBridge reads MAVLink from the local SITL TCP port and forwards to the Orchestrator via UDP over ZeroTier.
func StartBridge(sitlPort string, orchestratorIP string) {
	log.Printf("Starting Telemetry Bridge: SITL (TCP %s) -> Orchestrator (UDP %s:14550)", sitlPort, orchestratorIP)

	// 1. Connect to Orchestrator UDP target
	udpAddr, err := net.ResolveUDPAddr("udp", orchestratorIP+":14550")
	if err != nil {
		log.Fatalf("Failed to resolve orchestrator UDP address: %v", err)
	}

	udpConn, err := net.DialUDP("udp", nil, udpAddr)
	if err != nil {
		log.Fatalf("Failed to dial orchestrator UDP: %v", err)
	}
	defer udpConn.Close()

	// 2. Connect to ArduPilot SITL TCP stream
	var tcpConn net.Conn
	for {
		tcpConn, err = net.Dial("tcp", "127.0.0.1:"+sitlPort)
		if err == nil {
			break
		}
		log.Printf("Waiting for SITL on port %s...", sitlPort)
		time.Sleep(2 * time.Second)
	}
	defer tcpConn.Close()
	log.Println("Successfully connected to ArduPilot SITL!")

	// 3. Bridge the streams
	buffer := make([]byte, 1024)
	for {
		n, err := tcpConn.Read(buffer)
		if err != nil {
			log.Printf("Lost connection to SITL: %v", err)
			break
		}

		_, err = udpConn.Write(buffer[:n])
		if err != nil {
			log.Printf("Failed to forward UDP packet: %v", err)
		}
	}
}
