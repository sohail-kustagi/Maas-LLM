package client

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"maas_edge/pkg/config"
	"maas_edge/pkg/crypto"
)

func TestConnectToOrchestrator(t *testing.T) {
	// Mock Orchestrator Server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/edge/beacon" {
			t.Errorf("Expected path /v1/edge/beacon, got %s", r.URL.Path)
		}
		
		response := ClaimResponse{
			Status:       "provisioned",
			AssignedName: "Test_Drone",
			ZeroTierID:   "zt_12345",
		}
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	cfg := &config.Config{
		OrchestratorURL: server.URL,
		HardwareID:      "test_hw",
	}

	keys, _ := crypto.GenerateEphemeralKeys()
	
	claim, err := ConnectToOrchestrator(cfg, keys, "123456")
	if err != nil {
		t.Fatalf("Failed to connect: %v", err)
	}

	if claim.AssignedName != "Test_Drone" {
		t.Errorf("Expected Test_Drone, got %s", claim.AssignedName)
	}
}
