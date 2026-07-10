package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os/exec"

	"maas_edge/pkg/config"
	"maas_edge/pkg/crypto"
)

type BeaconRequest struct {
	HardwareID string `json:"hardware_id"`
	PairCode   string `json:"pair_code"`
	Status     string `json:"status"`
	Security   struct {
		EphemeralPublicKey string `json:"ephemeral_public_key"`
		DeviceSignature    string `json:"device_signature"`
	} `json:"security"`
}

type ClaimResponse struct {
	Status                    string `json:"status"`
	AssignedName              string `json:"assigned_name"`
	ZeroTierID                string `json:"zero_tier_id"`
	LiveKitURL                string `json:"livekit_url"`
	LiveKitJWT                string `json:"livekit_jwt"`
}

func ConnectToOrchestrator(cfg *config.Config, keys *crypto.KeyPair, pairCode string) (*ClaimResponse, error) {
	reqBody := BeaconRequest{
		HardwareID: cfg.HardwareID,
		PairCode:   pairCode,
		Status:     "awaiting_claim",
	}
	reqBody.Security.EphemeralPublicKey = keys.PublicKey
	reqBody.Security.DeviceSignature = "unsigned_for_now"

	jsonData, _ := json.Marshal(reqBody)

	url := fmt.Sprintf("%s/v1/edge/beacon", cfg.OrchestratorURL)
	log.Printf("Long-polling Orchestrator at %s with Pair Code: %s", url, pairCode)

	resp, err := http.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("orchestrator returned status: %d", resp.StatusCode)
	}

	var claim ClaimResponse
	if err := json.NewDecoder(resp.Body).Decode(&claim); err != nil {
		return nil, err
	}

	return &claim, nil
}

func JoinZeroTier(networkID string) error {
	log.Printf("Joining ZeroTier Network: %s", networkID)
	
	// Execute 'sudo zerotier-cli join <NetworkID>'
	// Note: in a real environment this requires the edge user to have sudo privileges for zerotier-cli
	cmd := exec.Command("sudo", "zerotier-cli", "join", networkID)
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		return fmt.Errorf("zerotier join failed: %v, output: %s", err, string(output))
	}
	
	log.Printf("ZeroTier Output: %s", string(output))
	return nil
}
