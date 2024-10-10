package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"time"

	"github.com/alitto/pond"
	"github.com/jessevdk/go-flags"
	"github.com/klauspost/compress/zstd"
	
"github.com/dgallion1/primeGen/logging"
	
)
var logIt logging.MyLogger

// generateStreams divides a huge number into ranges of maxPerWorker and spawns numWorkers to execute primesieve on each range.
func generateStreams(startValue, hugeNum, numWorkers, maxPerWorker int64) []chan map[string]interface{} {
	if numWorkers <= 0 || maxPerWorker <= 0 {
		panic("numWorkers and maxPerWorker must be greater than zero")
	}

	channels := make([]chan map[string]interface{}, numWorkers)
	var wg sync.WaitGroup

	
	// Calculate the range each worker should process.
	rangesPerWorker := hugeNum / numWorkers // Ceiling division
	logIt.Info().Msgf("Ranges per worker: %d hugeNum: %d", rangesPerWorker, hugeNum)
	for i := int64(0); i < numWorkers; i++ {
		start := startValue + i*rangesPerWorker
		end := start + rangesPerWorker
		if end > hugeNum {
			end = hugeNum
		}
		if start >= end {
			//fmt.Printf("Invalid range: [%d - %d]\n", start, end)
			//return channels
			continue
		}
		// Create a channel for each worker to output their primes
		channels[i] = make(chan map[string]interface{}, 1)
		//fmt.Println("Spawning worker for range: ", start, end)
		wg.Add(1)
		go func(ch chan map[string]interface{}, start, end int64) {
			defer wg.Done()
			primes := runPrimeSieve(start, end)
			ch <- map[string]interface{}{"start": start, "end": end, "primes": primes}
			close(ch)
		}(channels[i], start, end)
	}
	logIt.Info().Msgf("Total workers: %d", len(channels))
	// Wait for all goroutines to finish
	go func() {
		wg.Wait()
	}()

	return channels
}

// runPrimeSieve runs the primesieve command for the given range and returns the primes as a slice of int64.
func runPrimeSieve(start, end int64) []int64 {
	time.Sleep(time.Millisecond *time.Duration(rand.Int63n(2000)))
	logIt.Info().Msgf("Spawning worker for range: %d - %d", start, end)
	cmd := exec.Command("primesieve", fmt.Sprint(start), fmt.Sprint(end), "--print")
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		logIt.Error().Err(err).Msgf("Error creating stdout pipe: %v", err)
		return nil
	}

	if err := cmd.Start(); err != nil {
		logIt.Error().Msgf("Error starting command: %v", err)
		return nil
	}

	scanner := bufio.NewScanner(stdout)
	var primes = make([]int64, 0, 1000)
	delta := int64(0)
	// if scanner.Scan() {
	// 	line := scanner.Text()
	// 	if num, err := strconv.ParseInt(line, 10, 64); err == nil {
	// 		delta += num
	// 		primes = append(primes, num)
	// 	}
	// }

	for scanner.Scan() {
		line := scanner.Text()
		if num, err := strconv.ParseInt(line, 10, 64); err == nil {
			if len(primes) > 0 {
				delta += primes[len(primes)-1]
			}
			primes = append(primes, num-delta)
		}
	}

	if err := scanner.Err(); err != nil {
		logIt.Error().Err(err).Msgf("Error reading output: %v", err)
		return nil
	}

	if err := cmd.Wait(); err != nil {
		logIt.Error().Err(err).Msgf("Error waiting for command: %v", err)
		return nil
	}
	//fmt.Printf("Range [%d - %d] has %d primes\n", start, end, len(primes))
	return primes
}

// generateHugeStreams feeds huge numbers in order to generateStreams
func generateHugeStreams(initialStart, stepSize, end, numWorkers, maxPerWorker int64) {
	os.Mkdir("out", 0755)
	pool := pond.New(5, 20)

	for i := int64(initialStart); i*stepSize < end; i++ {
		startValue := initialStart + i*stepSize
		hugeNum := startValue + stepSize
		logIt.Info().Msgf("Generating streams for range [%d - %d]", startValue, hugeNum)
		channels := generateStreams(startValue, hugeNum, numWorkers, maxPerWorker)

		for j, ch := range channels {
			if ch == nil {
				continue
			}
			//fmt.Printf("Worker %d: \n", j)
			for result := range ch {
				result := result
				pool.Submit(func() {
					time.Sleep(time.Millisecond *time.Duration(rand.Int63n(1000)))
					start := result["start"].(int64)
					end := result["end"].(int64)
					primes := result["primes"].([]int64)

					filename := fmt.Sprintf("out/%d-%d.txt", start, end)
					file, err := os.Create(filename)
					if err != nil {
						logIt.Error().Err(err).Msgf("Error creating file %s: %v", filename, err)
						return
					}
					defer file.Close()
					if true {
						zstdWriter, err := zstd.NewWriter(file)
						if err != nil {
							logIt.Error().Err(err).Msgf("Error creating zstd writer for file %s: %v", filename, err)
							return
						}
						defer zstdWriter.Close()
						// Use binary.Write to write the array to the zstd writer
						err = binary.Write(zstdWriter, binary.LittleEndian, primes)
						if err != nil {
							logIt.Error().Err(err).Msgf("Error writing to file %s: %v", filename, err)
							return
						}

						// Flush the zstd writer to ensure all data is written
						err = zstdWriter.Flush()
						if err != nil {
							logIt.Error().Err(err).Msgf("Error writing to file %s: %v", filename, err)
							return
						}

					} else {
						file.Write([]byte(fmt.Sprintf("%d\n", len(primes))))
						file.Write([]byte(fmt.Sprintf("%v\n", primes)))

					}
					logIt.Info().Msgf("Worker %d: Range [%d - %d] saved to %s", j, start, end, filename)
				})
			}
		}
	}
	pool.StopAndWait()
}

func main() {
	os.Mkdir("logs", 0755)
	
 	logIt = logging.Configure(logging.Config{
		ConsoleLoggingEnabled: true,
		FileLoggingEnabled:    true,
		EncodeLogsAsJson:      true,
		Directory:             "./logs",
		Filename:              "primeGen.log",
		MaxSize:               1000000, // 1MB
		MaxBackups:            3,
		MaxAge:                28, // days
	})

	var opts struct {
		InitialStart int64 `short:"i" long:"initialStart" description:"Initial start value" default:"0"`
		StepSize     int64 `short:"s" long:"stepSize" description:"Step size for each iteration" default:"100000000000"`
		NumWorkers   int64 `short:"n" long:"numWorkers" description:"Number of workers" default:"10"`
		MaxPerWorker int64 `short:"m" long:"maxPerWorker" description:"Maximum range per worker" default:"100000000000"`
		End          int64 `short:"t" long:"End" description:"End point" default:"1000000000000000"`
	}

	_, err := flags.Parse(&opts)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	generateHugeStreams(opts.InitialStart, opts.StepSize, opts.End, opts.NumWorkers, opts.MaxPerWorker)
}
