# twitch_chat.py
import socket
import time
import logging
from threading import Lock
from config import TWITCH_CHANNEL, TWITCH_BOT_USERNAME, TWITCH_OAUTH_TOKEN

logger = logging.getLogger(__name__)

class TwitchBot:
    def __init__(self):
        self.server = 'irc.chat.twitch.tv'
        self.port = 6667
        self.channel = f'#{TWITCH_CHANNEL}'
        self.username = TWITCH_BOT_USERNAME
        self.token = TWITCH_OAUTH_TOKEN
        self.sock = None
        self.connected = False
        self.connection_lock = Lock()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_connection_attempt = 0
        self.connection_timeout = 10
        
    def connect(self):
        """Connect to Twitch IRC with retry logic"""
        with self.connection_lock:
            if self.connected and self.sock:
                return True
                
            current_time = time.time()
            
            # Rate limit connection attempts
            if current_time - self.last_connection_attempt < 30:
                logger.warning("Connection attempt rate limited")
                return False
                
            self.last_connection_attempt = current_time
            
            try:
                # Close existing socket if present
                if self.sock:
                    try:
                        self.sock.close()
                    except:
                        pass
                        
                # Create new socket
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.connection_timeout)
                
                # Connect to Twitch IRC
                logger.info(f"Connecting to {self.server}:{self.port}...")
                self.sock.connect((self.server, self.port))
                
                # Authentication sequence
                self.sock.send(f"PASS {self.token}\r\n".encode('utf-8'))
                self.sock.send(f"NICK {self.username}\r\n".encode('utf-8'))
                self.sock.send(f"JOIN {self.channel}\r\n".encode('utf-8'))
                
                # Wait for connection confirmation
                response = self.sock.recv(2048).decode('utf-8', errors='ignore')
                
                if "Welcome" in response or "001" in response or "353" in response:
                    self.connected = True
                    self.reconnect_attempts = 0
                    logger.info(f"Successfully connected to Twitch as {self.username} in {self.channel}")
                    return True
                else:
                    logger.warning(f"Unexpected connection response: {response}")
                    self.connected = False
                    return False
                    
            except socket.timeout:
                logger.error("Connection timeout")
                self.connected = False
                return False
            except socket.gaierror as e:
                logger.error(f"DNS resolution failed: {e}")
                self.connected = False
                return False
            except ConnectionRefusedError:
                logger.error("Connection refused by Twitch")
                self.connected = False
                return False
            except Exception as e:
                logger.error(f"Connection error: {e}")
                self.connected = False
                return False
            finally:
                if not self.connected and self.sock:
                    try:
                        self.sock.close()
                    except:
                        pass
                    self.sock = None

    def reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return False
            
        self.reconnect_attempts += 1
        backoff_time = min(60, 2 ** self.reconnect_attempts)
        
        logger.info(f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} in {backoff_time}s")
        time.sleep(backoff_time)
        
        return self.connect()

    def is_connected(self):
        """Check if bot is currently connected"""
        if not self.connected or not self.sock:
            return False
            
        try:
            # Send a minimal ping to test connection
            self.sock.settimeout(1)
            self.sock.send(b"PING :tmi.twitch.tv\r\n")
            return True
        except:
            self.connected = False
            return False
        finally:
            if self.sock:
                try:
                    self.sock.settimeout(None)
                except:
                    pass

    def send_message(self, message):
        """Send message with automatic reconnection on failure"""
        if not message or len(message.strip()) == 0:
            logger.warning("Attempted to send empty message")
            return False
            
        # Truncate long messages
        if len(message) > 500:
            message = message[:497] + "..."
            
        max_attempts = 2
        
        for attempt in range(max_attempts):
            if not self.is_connected():
                logger.warning("Not connected, attempting to reconnect...")
                if not self.reconnect():
                    continue
                    
            try:
                with self.connection_lock:
                    if not self.sock:
                        continue
                        
                    self.sock.settimeout(5)
                    message_bytes = f"PRIVMSG {self.channel} :{message}\r\n".encode('utf-8')
                    self.sock.send(message_bytes)
                    
                    logger.info(f"Sent: {message}")
                    return True
                    
            except socket.timeout:
                logger.error("Message send timeout")
                self.connected = False
            except socket.error as e:
                logger.error(f"Socket error during send: {e}")
                self.connected = False
            except Exception as e:
                logger.error(f"Unexpected error sending message: {e}")
                self.connected = False
            finally:
                if self.sock:
                    try:
                        self.sock.settimeout(None)
                    except:
                        pass
        
        logger.error("Failed to send message after all attempts")
        return False

    def listen(self, callback=None):
        """Listen for incoming messages (optional feature)"""
        if not self.is_connected():
            return
            
        try:
            self.sock.settimeout(1)
            while self.connected:
                try:
                    data = self.sock.recv(2048)
                    if not data:
                        break
                        
                    message = data.decode('utf-8', errors='ignore')
                    
                    # Handle PING requests
                    if message.startswith('PING'):
                        self.sock.send(b"PONG :tmi.twitch.tv\r\n")
                        continue
                        
                    if callback:
                        callback(message)
                        
                except socket.timeout:
                    continue  # Normal timeout, keep listening
                except socket.error:
                    break
                    
        except Exception as e:
            logger.error(f"Error in listen loop: {e}")
        finally:
            self.connected = False

    def disconnect(self):
        """Clean disconnection"""
        with self.connection_lock:
            self.connected = False
            if self.sock:
                try:
                    self.sock.send(f"PART {self.channel}\r\n".encode('utf-8'))
                    self.sock.close()
                except:
                    pass
                finally:
                    self.sock = None
                    
            logger.info("Disconnected from Twitch")

    def __del__(self):
        """Cleanup on object destruction"""
        self.disconnect()
