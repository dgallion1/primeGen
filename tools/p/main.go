package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

func main() {
	// Pre-allocate a buffer to avoid GC overhead
	var data []int64
	for {
		// Read length (8 bytes, 64 bits) from stdin
		var length int64
		err := binary.Read(os.Stdin, binary.LittleEndian, &length)
		if err == io.EOF {
			break
		} else if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading length from stdin: %v\n", err)
			os.Exit(1)
		}

		// Resize the slice to hold the required data
		if int64(cap(data)) < length {
			data = make([]int64, length)
		} else {
			data = data[:length]
		}

		// Read all data into the slice
		err = binary.Read(os.Stdin, binary.LittleEndian, &data)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading data from stdin: %v\n", err)
			os.Exit(1)
		}

		// Print all data
		for _, value := range data {
			//fmt.Println(value)
			_ = value
		}
	}
} 
