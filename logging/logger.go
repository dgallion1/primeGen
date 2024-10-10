package logging

import (
	"fmt"
	"strings"
	"time"

	"github.com/rs/zerolog"
	//"github.com/rs/zerolog/log"
	"io"
	"os"
	"path"

	"gopkg.in/natefinch/lumberjack.v2"
)

// Configuration for logging
type Config struct {
	// Enable console logging
	ConsoleLoggingEnabled bool

	// EncodeLogsAsJson makes the log framework log JSON
	EncodeLogsAsJson bool
	// FileLoggingEnabled makes the framework log to a file
	// the fields below can be skipped if this value is false!
	FileLoggingEnabled bool
	// Directory to log to to when filelogging is enabled
	Directory string
	// Filename is the name of the logfile which will be placed inside the directory
	Filename string
	// MaxSize the max size in MB of the logfile before it's rolled
	MaxSize int
	// MaxBackups the max number of rolled files to keep
	MaxBackups int
	// MaxAge the max age in days to keep a logfile
	MaxAge int
}

type MyLogger struct {
	zerolog.Logger
}

var LogIt MyLogger

func NewLogger() MyLogger {
	// create output configuration
	output := zerolog.ConsoleWriter{Out: os.Stdout, TimeFormat: time.RFC3339}

	// Format level: fatal, error, debug, info, warn
	output.FormatLevel = func(i interface{}) string {
		return strings.ToUpper(fmt.Sprintf("| %-6s|", i))
	}
	output.FormatFieldName = func(i interface{}) string {
		return fmt.Sprintf("%s:", i)
	}
	output.FormatFieldValue = func(i interface{}) string {
		return fmt.Sprintf("%s", i)
	}

	// format error
	output.FormatErrFieldName = func(i interface{}) string {
		return fmt.Sprintf("%s: ", i)
	}

	zerolog := zerolog.New(output).With().Timestamp().Logger()
	LogIt = MyLogger{zerolog}
	return LogIt
}

func (l *MyLogger) LogInfo() *zerolog.Event {
	return l.Logger.Info()
}

func (l *MyLogger) LogError() *zerolog.Event {
	return l.Logger.Error()
}

func (l *MyLogger) LogDebug() *zerolog.Event {
	return l.Logger.Debug()
}

func (l *MyLogger) LogWarn() *zerolog.Event {
	return l.Logger.Warn()
}

func (l *MyLogger) LogFatal() *zerolog.Event {
	return l.Logger.Fatal()
}

// Configure sets up the logging framework
//
// In production, the container logs will be collected and file logging should be disabled. However,
// during development it's nicer to see logs as text and optionally write to a file when debugging
// problems in the containerized pipeline
//
// The output log file will be located at /var/log/service-xyz/service-xyz.log and
// will be rolled according to configuration set.
func Configure(config Config) MyLogger {
	var writers []io.Writer

	if config.ConsoleLoggingEnabled {
		writers = append(writers, zerolog.ConsoleWriter{Out: os.Stderr})
	}
	if config.FileLoggingEnabled {
		writers = append(writers, newRollingFile(config))
	}
	mw := io.MultiWriter(writers...)

	// zerolog.SetGlobalLevel(zerolog.DebugLevel)
	logger := zerolog.New(mw).With().Caller().Timestamp().Logger()
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	logger.Info().
		Bool("fileLogging", config.FileLoggingEnabled).
		Bool("jsonLogOutput", config.EncodeLogsAsJson).
		Str("logDirectory", config.Directory).
		Str("fileName", config.Filename).
		Int("maxSizeMB", config.MaxSize).
		Int("maxBackups", config.MaxBackups).
		Int("maxAgeInDays", config.MaxAge).
		Msg("logging configured")

	return MyLogger{
		Logger: logger,
	}
}

func newRollingFile(config Config) io.Writer {
	if err := os.MkdirAll(config.Directory, 0744); err != nil {
		log := zerolog.New(os.Stderr).With().Timestamp().Logger()
		log.Error().Err(err).Str("path", config.Directory).Msg("can't create log directory")
		return nil
	}

	return &lumberjack.Logger{
		Filename:   path.Join(config.Directory, config.Filename),
		MaxBackups: config.MaxBackups, // files
		MaxSize:    config.MaxSize,    // megabytes
		MaxAge:     config.MaxAge,     // days
	}
}
