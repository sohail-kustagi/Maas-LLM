package portal

import (
	_ "embed"
	"log"
	"net/http"
	"strings"
)

//go:embed ui/index.html
var portalHTML string

func StartCaptivePortal(pairCode string, port string) {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")

		// Inject the dynamically generated pair code into the HTML
		finalHTML := strings.Replace(portalHTML, "{{PAIR_CODE}}", pairCode, 1)
		w.Write([]byte(finalHTML))
	})

	log.Printf("Starting Captive Portal on port %s", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Printf("Captive portal shut down: %v", err)
	}
}
