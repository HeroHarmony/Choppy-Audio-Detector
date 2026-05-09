# twitch_chat.py
import socket
import time
import logging
from threading import Lock
try:
    from config import TWITCH_CHANNEL, TWITCH_BOT_USERNAME, TWITCH_OAUTH_TOKEN
except Exception:
    TWITCH_CHANNEL = ""
    TWITCH_BOT_USERNAME = ""
    TWITCH_OAUTH_TOKEN = ""

logger = logging.getLogger(__name__)

class TwitchBot:
    def __init__(self, channel=None, username=None, token=None):
        self.server = 'irc.chat.twitch.tv'
        self.port = 6667
        channel_name = (channel or TWITCH_CHANNEL or "").strip().lstrip("#")
        self.channel = f'#{channel_name}'
        self.username = username or TWITCH_BOT_USERNAME
        self.token = token or TWITCH_OAUTH_TOKEN
        self.sock = None
        self.connected = False
        self.connection_lock = Lock()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_connection_attempt = 0
        self.connection_timeout = 10
        self.last_error = ""
        self.last_response = ""
        
    def connect(self, deadline_monotonic=None):
        """Connect to Twitch IRC with retry logic"""
        with self.connection_lock:
            if self.connected and self.sock:
                return True
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                self.last_error = "Connection attempt skipped: send window expired"
                logger.warning(self.last_error)
                return False
                
            current_time = time.time()
            
            # Rate limit connection attempts
            if current_time - self.last_connection_attempt < 30:
                self.last_error = "Connection attempt rate limited"
                logger.warning(self.last_error)
                return False
                
            self.last_connection_attempt = current_time
            
            try:
                if not str(self.username or "").strip():
                    self.last_error = "Missing Twitch username."
                    logger.warning(self.last_error)
                    self.connected = False
                    return False
                if not str(self.token or "").strip():
                    self.last_error = "Missing Twitch OAuth token."
                    logger.warning(self.last_error)
                    self.connected = False
                    return False
                if not str(self.channel or "").strip() or str(self.channel).strip() == "#":
                    self.last_error = "Missing Twitch channel."
                    logger.warning(self.last_error)
                    self.connected = False
                    return False

                # Close existing socket if present
                if self.sock:
                    try:
                        self.sock.close()
                    except:
                        pass
                        
                # Resolve and connect using any available address family (IPv4/IPv6).
                timeout_seconds = self.connection_timeout
                if deadline_monotonic is not None:
                    remaining = deadline_monotonic - time.monotonic()
                    if remaining <= 0:
                        self.last_error = "Connection timeout window expired"
                        logger.warning(self.last_error)
                        self.connected = False
                        return False
                    timeout_seconds = max(0.5, min(timeout_seconds, remaining))
                resolved = socket.getaddrinfo(self.server, self.port, socket.AF_UNSPEC, socket.SOCK_STREAM)
                if not resolved:
                    self.last_error = f"DNS resolution returned no addresses for {self.server}"
                    logger.error(self.last_error)
                    self.connected = False
                    return False

                connection_errors = []
                self.sock = None
                for family, socktype, proto, _canonname, sockaddr in resolved:
                    candidate = socket.socket(family, socktype, proto)
                    candidate.settimeout(timeout_seconds)
                    try:
                        candidate.connect(sockaddr)
                        self.sock = candidate
                        break
                    except Exception as connect_exc:
                        connection_errors.append(f"{sockaddr}: {connect_exc}")
                        try:
                            candidate.close()
                        except Exception:
                            pass

                if self.sock is None:
                    self.last_error = "Unable to connect to Twitch IRC address: " + "; ".join(connection_errors)
                    logger.error(self.last_error)
                    self.connected = False
                    return False

                # Connect to Twitch IRC
                logger.info(f"Connecting to {self.server}:{self.port}...")
                
                # Authentication sequence
                self.sock.send(b"CAP REQ :twitch.tv/tags twitch.tv/commands\r\n")
                self.sock.send(f"PASS {self.token}\r\n".encode('utf-8'))
                self.sock.send(f"NICK {self.username}\r\n".encode('utf-8'))
                self.sock.send(f"JOIN {self.channel}\r\n".encode('utf-8'))
                
                # Wait for connection confirmation. Twitch may send CAP ACK before
                # the numeric welcome/join messages, so keep reading briefly.
                response_parts = []
                deadline = time.time() + timeout_seconds
                connected_response = ""
                auth_failed_response = ""

                while time.time() < deadline:
                    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                        break
                    response = self.sock.recv(2048).decode('utf-8', errors='ignore')
                    if not response:
                        break
                    response_parts.append(response)
                    combined_response = "".join(response_parts)
                    self.last_response = combined_response

                    if (
                        "Login authentication failed" in combined_response
                        or "Improperly formatted auth" in combined_response
                        or "Error logging in" in combined_response
                        or "NOTICE * :Login" in combined_response
                    ):
                        auth_failed_response = combined_response
                        break

                    if (
                        "Welcome" in combined_response
                        or " 001 " in combined_response
                        or " 353 " in combined_response
                        or " JOIN " in combined_response
                    ):
                        connected_response = combined_response
                        break

                if connected_response:
                    self.connected = True
                    self.reconnect_attempts = 0
                    logger.info(f"Successfully connected to Twitch as {self.username} in {self.channel}")
                    return True
                elif auth_failed_response:
                    self.last_error = f"Twitch authentication failed: {auth_failed_response.strip()}"
                    logger.warning(self.last_error)
                    self.connected = False
                    return False
                else:
                    response_text = self.last_response.strip() or "No response before timeout"
                    self.last_error = f"Unexpected connection response: {response_text}"
                    logger.warning(self.last_error)
                    self.connected = False
                    return False
                    
            except socket.timeout:
                self.last_error = "Connection timeout"
                logger.error(self.last_error)
                self.connected = False
                return False
            except socket.gaierror as e:
                self.last_error = f"DNS resolution failed: {e}"
                logger.error(self.last_error)
                self.connected = False
                return False
            except ConnectionRefusedError:
                self.last_error = "Connection refused by Twitch"
                logger.error(self.last_error)
                self.connected = False
                return False
            except Exception as e:
                self.last_error = f"Connection error: {e}"
                logger.error(self.last_error)
                self.connected = False
                return False
            finally:
                if not self.connected and self.sock:
                    try:
                        self.sock.close()
                    except:
                        pass
                    self.sock = None

    def reconnect(self, deadline_monotonic=None):
        """Attempt to reconnect with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return False
            
        self.reconnect_attempts += 1
        backoff_time = min(60, 2 ** self.reconnect_attempts)
        if deadline_monotonic is not None:
            remaining = deadline_monotonic - time.monotonic()
            if remaining <= 0:
                logger.warning("Reconnect skipped: send window expired")
                return False
            backoff_time = min(backoff_time, remaining)
        
        logger.info(f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} in {backoff_time}s")
        time.sleep(backoff_time)
        
        return self.connect(deadline_monotonic=deadline_monotonic)

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

    def send_message(self, message, max_total_seconds=None):
        """Send message with automatic reconnection on failure"""
        if not message or len(message.strip()) == 0:
            logger.warning("Attempted to send empty message")
            return False
            
        # Truncate long messages
        if len(message) > 500:
            message = message[:497] + "..."
        deadline_monotonic = None
        if max_total_seconds is not None:
            try:
                deadline_monotonic = time.monotonic() + max(0.1, float(max_total_seconds))
            except Exception:
                deadline_monotonic = None
            
        max_attempts = 2
        
        for attempt in range(max_attempts):
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                logger.warning("Send window expired before message could be delivered")
                return False
            if not self.is_connected():
                logger.warning("Not connected, attempting to reconnect...")
                if not self.reconnect(deadline_monotonic=deadline_monotonic):
                    continue
                    
            try:
                with self.connection_lock:
                    if not self.sock:
                        continue
                        
                    timeout_seconds = 5
                    if deadline_monotonic is not None:
                        remaining = deadline_monotonic - time.monotonic()
                        if remaining <= 0:
                            logger.warning("Send window expired before socket send")
                            return False
                        timeout_seconds = max(0.2, min(timeout_seconds, remaining))
                    self.sock.settimeout(timeout_seconds)
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

    def listen(self, callback=None, should_continue=None):
        """Listen for incoming messages (optional feature)"""
        if not self.is_connected():
            return
            
        try:
            self.sock.settimeout(1)
            while self.connected and (should_continue is None or should_continue()):
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
