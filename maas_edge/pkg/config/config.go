package config

import (
	"log"
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	OrchestratorURL string
	SITLPort        string
	HardwareID      string
}

func LoadConfig() *Config {
	err := godotenv.Load()
	if err != nil {
		log.Println("No .env file found, relying on system environment variables")
	}

	sitlPort := os.Getenv("SITL_PORT")
	if sitlPort == "" {
		sitlPort = "5760" // Default ArduPilot TCP port
	}

	orchestratorURL := os.Getenv("ORCHESTRATOR_URL")
	if orchestratorURL == "" {
		orchestratorURL = "http://localhost:8080"
	}

	hardwareID := os.Getenv("HARDWARE_ID")
	if hardwareID == "" {
		hardwareID = "edge_device_default"
	}

	return &Config{
		OrchestratorURL: orchestratorURL,
		SITLPort:        sitlPort,
		HardwareID:      hardwareID,
	}
}
