package crypto

import (
	"testing"
)

func TestGenerateEphemeralKeys(t *testing.T) {
	keys, err := GenerateEphemeralKeys()
	if err != nil {
		t.Fatalf("Failed to generate keys: %v", err)
	}

	if keys.PublicKey == "" {
		t.Errorf("Public key is empty")
	}

	if keys.PrivateKey == nil {
		t.Errorf("Private key is nil")
	}
}

func TestGeneratePairCode(t *testing.T) {
	code := GeneratePairCode()
	if len(code) != 6 {
		t.Errorf("Expected pair code length 6, got %d", len(code))
	}
}
