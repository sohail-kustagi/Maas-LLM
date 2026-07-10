package main

import (
	"log"
	"time"

	"maas_edge/internal/client"
	"maas_edge/internal/portal"
	"maas_edge/internal/telemetry"
	"maas_edge/internal/video"
	"maas_edge/pkg/config"
	"maas_edge/pkg/crypto"
)

func main() {
	cfg := config.LoadConfig()

	// 1. Generate Zero-Trust Keys & Pair Code
	keys, err := crypto.GenerateEphemeralKeys()
	if err != nil {
		log.Fatalf("Failed to generate keys: %v", err)
	}
	pairCode := crypto.GeneratePairCode()

	// 2. Start Captive Portal (non-blocking)
	go portal.StartCaptivePortal(pairCode, "8080")

	// 3. Connect to Orchestrator (Long-polling)
	claim, err := client.ConnectToOrchestrator(cfg, keys, pairCode)
	if err != nil {
		log.Fatalf("Failed to provision drone: %v", err)
	}

	log.Printf("Successfully Provisioned! Name: %s", claim.AssignedName)

	// 4. Join ZeroTier Network
	err = client.JoinZeroTier(claim.ZeroTierID)
	if err != nil {
		log.Printf("Warning: ZeroTier join failed: %v", err)
	} else {
		log.Println("Successfully joined ZeroTier network!")
	}
	
	// Wait a moment for the ZeroTier virtual adapter to get its IP
	time.Sleep(3 * time.Second)

	// 5. Start Telemetry Bridge (SITL TCP -> ZeroTier UDP)
	// We extract the orchestrator IP from the URL (simplified for now, assume localhost or IP)
	// In production, the orchestrator should return its ZeroTier IP in the claim.
	go telemetry.StartBridge(cfg.SITLPort, "127.0.0.1") // Hardcoded local for SITL testing without real ZeroTier IPs

	// 6. Start Video Publisher (Webcam -> LiveKit)
	err = video.StartVideoPublisher(claim.LiveKitURL, claim.LiveKitJWT)
	if err != nil {
		log.Fatalf("Video Publisher failed: %v", err)
	}
}
