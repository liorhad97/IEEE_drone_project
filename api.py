from data_models.sensor_data_model import SensorsData
import threading
# You will need to install sseclient-py and requests:
# pip install sseclient-py requests
try:
    from sseclient import SSEClient
    import requests # sseclient often relies on the requests library for HTTP connections
except ImportError:
    SSEClient = None
    requests = None
    # A more detailed warning will be printed by the API class __init__ if libraries are missing.

class API:
    def __init__(self, stream_url: str = "YOUR_SSE_STREAM_URL_HERE_PLEASE_REPLACE"):
        self._objective_found = False
        self.stream_url = stream_url
        self._stop_event = threading.Event()  # Used to signal the listening thread to stop
        self.listener_thread = None

        if not SSEClient or not requests:
            print("CRITICAL API ERROR: 'sseclient-py' or 'requests' library not found.")
            print("Objective status updates via server stream will NOT work.")
            print("Please install them by running: pip install sseclient-py requests")
            return  # Exit __init__ early if essential libraries are missing

        if self.stream_url == "YOUR_SSE_STREAM_URL_HERE_PLEASE_REPLACE" or not self.stream_url:
            print("API Init Warning: SSE stream URL is not configured or is invalid.")
            print("Objective status will not update from a server stream.")
        else:
            self.listener_thread = threading.Thread(target=self._listen_to_stream, daemon=True)
            self.listener_thread.start()
            print(f"API Init: Listener thread started for SSE stream at {self.stream_url}")

    def _listen_to_stream(self):
        """
        Listens to an SSE stream for notifications about the objective.
        Updates self._objective_found based on received messages.
        This method is intended to run in a separate thread.
        """
        # This check is redundant if __init__ exits early, but good for safety if _listen_to_stream was called otherwise
        if not SSEClient or not requests:
            print("SSE Listener Error: SSE client libraries not available. Cannot listen to stream.")
            return

        print(f"SSE Listener: Attempting to connect to stream at {self.stream_url}...")
        try:
            # Note: SSEClient can take various requests options, e.g., for headers, timeouts.
            # messages = SSEClient(self.stream_url, timeout=30) # Example with timeout
            messages = SSEClient(self.stream_url)
            for msg in messages:
                if self._stop_event.is_set():
                    print("SSE Listener: Stop event received, exiting listener loop.")
                    break
                
                # IMPORTANT: Customize the following logic based on your server's SSE message format.
                # This is a placeholder for how you might process messages.
                # Log the raw message for debugging purposes when setting up:
                print(f"SSE Listener: Received message - Event: '{msg.event}', Data: '{msg.data}'")
                
                # Example: Server sends an event named "objective_status"
                # and the data might be a JSON string like '{"status": "found"}' or '{"status": "reset"}'
                # Or, it might be a simpler event name like "objective_found_event".
                if msg.event == "objective_found_event":  # Replace "objective_found_event" with your actual event name
                    print("SSE Listener: 'objective_found_event' received. Setting objective found to True.")
                    self._objective_found = True
                    # Optional: If the objective, once found, means the task is done, you might stop listening.
                    # self._stop_event.set() # This would cause the listener thread to exit.
                    # break
                elif msg.event == "objective_reset_event":  # Example for an event that resets the status
                    print("SSE Listener: 'objective_reset_event' received. Setting objective found to False.")
                    self._objective_found = False
                # Add more conditions here if your server sends other relevant events or data structures.

        except requests.exceptions.ConnectionError as e:
            print(f"SSE Listener: Connection error to {self.stream_url}: {e}. Check server, network, and URL. Will not auto-retry in this example.")
        except Exception as e:
            # Catching a broad exception to prevent the thread from crashing silently and to log the error.
            print(f"SSE Listener: An unexpected error occurred: {e}")
        finally:
            if 'messages' in locals() and hasattr(messages, 'close'):
                messages.close() # Ensure the SSEClient connection is closed
            print(f"SSE Listener: Thread for stream {self.stream_url} has finished.")

    def get_drone_data(self) -> SensorsData: # Now an instance method
        # ... implementation for get_drone_data ...
        # This method was part of the original class.
        # Ensure its logic is preserved or updated as needed.
        # For now, it's a placeholder.
        print("API: get_drone_data() called (implementation pending).")
        # Example: return SensorsData(...)
        pass

    def is_drone_find_objective(self) -> bool: # Now an instance method
        """
        Returns True if the drone has found the objective, False otherwise.
        The status is updated by a background listener to a server stream.
        """
        return self._objective_found

    def stop_listening(self):
        """Stops the SSE listener thread gracefully."""
        if self.listener_thread and self.listener_thread.is_alive():
            print("API: Signaling SSE listener thread to stop...")
            self._stop_event.set()
            self.listener_thread.join(timeout=5)  # Wait for the thread to finish
            if self.listener_thread.is_alive():
                print("API Warning: SSE listener thread did not stop in time.")
            else:
                print("API: SSE listener thread stopped successfully.")
        else:
            print("API: SSE listener thread not running or already stopped.")

    # Ensure cleanup when the API object is deleted
    def __del__(self):
        print("API object is being deleted. Ensuring listener is stopped.")
        self.stop_listening()

