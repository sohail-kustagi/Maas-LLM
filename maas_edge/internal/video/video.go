package video

import (
	"log"

	lksdk "github.com/livekit/server-sdk-go"
	"github.com/livekit/server-sdk-go/pkg/samplebuilder"
	"github.com/pion/webrtc/v3"
	"github.com/pion/webrtc/v3/pkg/media"
	"github.com/pion/mediadevices"
	"github.com/pion/mediadevices/pkg/codec/x264"
	_ "github.com/pion/mediadevices/pkg/driver/camera" // This is required to register camera adapter
)

func StartVideoPublisher(url string, token string) error {
	log.Printf("Connecting to LiveKit Room at %s", url)

	room, err := lksdk.ConnectToRoomWithToken(url, token, lksdk.WithAutoSubscribe(false))
	if err != nil {
		return err
	}
	defer room.Disconnect()

	log.Println("Connected to LiveKit. Attempting to capture webcam...")

	x264Params, err := x264.NewParams()
	if err != nil {
		return err
	}
	x264Params.BitRate = 1000000 // 1 Mbps

	codecSelector := mediadevices.NewCodecSelector(
		mediadevices.WithVideoEncoders(&x264Params),
	)

	mediaStream, err := mediadevices.GetUserMedia(mediadevices.MediaStreamConstraints{
		Video: func(c *mediadevices.VideoTrackConstraints) {
			c.Codec = mediadevices.ExactParameter(webrtc.MimeTypeH264)
		},
		Codec: codecSelector,
	})

	if err != nil {
		return err
	}

	videoTrack := mediaStream.GetVideoTracks()[0]

	// Publish track to LiveKit
	localTrack, err := lksdk.NewLocalTrack(videoTrack.(*webrtc.TrackLocalStaticSample))
	if err != nil {
		return err
	}

	publication, err := room.LocalParticipant.PublishTrack(localTrack, &lksdk.TrackPublicationOptions{
		Name: "drone_camera",
	})
	if err != nil {
		return err
	}

	log.Printf("Successfully published webcam track to LiveKit: %s", publication.SID())

	// Keep alive
	select {}
}
