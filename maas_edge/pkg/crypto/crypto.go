package crypto

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"math/big"
	"fmt"
)

type KeyPair struct {
	PublicKey  string
	PrivateKey ed25519.PrivateKey
}

func GenerateEphemeralKeys() (*KeyPair, error) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, err
	}

	return &KeyPair{
		PublicKey:  base64.StdEncoding.EncodeToString(pub),
		PrivateKey: priv,
	}, nil
}

func GeneratePairCode() string {
	max := big.NewInt(999999)
	n, _ := rand.Int(rand.Reader, max)
	return fmt.Sprintf("%06d", n.Int64())
}
