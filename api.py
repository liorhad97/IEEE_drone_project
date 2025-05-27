from data_models.sensor_data_model import SensorsData
import threading
import time
import random # For simulation
import torch # For simulation data

# Conditional import for SSEClient and requests
try:
    from sseclient import SSEClient
    import requests
except ImportError:
    SSEClient = None
    requests = None

class API:
    def __init__(self, 
                 stream_url: str = "YOUR_SSE_STREAM_URL_HERE_PLEASE_REPLACE", 
                 simulate: bool = True,
                 # Simulation specific parameters with defaults
                 sim_objective_find_interval_seconds: float = 15.0,
                 sim_objective_reset_interval_seconds: float = 15.0, # Time after found to reset
                 sim_listener_tick_seconds: float = 0.5,
                 sim_initial_gps: list[float] = None, # Default [32.0853, 34.7818, 10.0]
                 sim_gps_drift_scale: float = 0.0001,
                 sim_accel_range: tuple[float, float] = (-0.5, 0.5),
                 sim_gyro_range: tuple[float, float] = (-0.1, 0.1),
                 sim_teta_yaw_range: tuple[float, float] = (-3.14, 3.14), # Full circle for yaw
                 sim_teta_pitch_range: tuple[float, float] = (-1.57, 1.57) # Half circle for pitch
                ):
        self._objective_found = False
        self.stream_url = stream_url
        self._stop_event = threading.Event()
        self.listener_thread = None
        self.sse_client_instance = None
        self.simulate = simulate

        # Store simulation parameters
        self.sim_objective_find_interval = sim_objective_find_interval_seconds
        self.sim_objective_reset_interval = sim_objective_reset_interval_seconds
        self.sim_listener_tick = sim_listener_tick_seconds
        self.sim_gps_center = sim_initial_gps if sim_initial_gps is not None else [32.0853, 34.7818, 10.0]
        self._current_simulated_gps = list(self.sim_gps_center) # Mutable copy
        self.sim_gps_drift_scale = sim_gps_drift_scale
        self.sim_accel_range = sim_accel_range
        self.sim_gyro_range = sim_gyro_range
        self.sim_teta_yaw_range = sim_teta_yaw_range
        self.sim_teta_pitch_range = sim_teta_pitch_range
        
        self._simulation_last_event_time = time.time()
        self._sim_objective_currently_found = False


        if self.simulate:
            print(f"API Init: Running in SIMULATION mode with tick interval {self.sim_listener_tick}s.")
            self.listener_thread = threading.Thread(target=self._simulated_listen_to_stream, daemon=True)
            self.listener_thread.start()
            print("API Init: Simulated listener thread started.")
        else:
            print(f"API Init: Running in LIVE mode. Attempting to connect to SSE stream at {self.stream_url}")
            if not SSEClient or not requests:
                print("CRITICAL API ERROR: 'sseclient-py' or 'requests' library not found for LIVE mode.")
                print("Objective status updates via server stream will NOT work.")
                print("Please install them by running: pip install sseclient-py requests")
                return

            if self.stream_url == "YOUR_SSE_STREAM_URL_HERE_PLEASE_REPLACE" or not self.stream_url:
                print("API Init Warning: SSE stream URL is not configured or is invalid for LIVE mode.")
                print("Objective status will not update from a server stream.")
            else:
                self.listener_thread = threading.Thread(target=self._live_listen_to_stream, daemon=True)
                self.listener_thread.start()
                print(f"API Init: Live listener thread started for SSE stream at {self.stream_url}")


    def _simulated_listen_to_stream(self):
        """Simulates receiving SSE events to toggle objective status based on time intervals."""
        print(f"Simulated SSE Listener: Starting. Find after {self.sim_objective_find_interval}s, Reset after another {self.sim_objective_reset_interval}s.")
        self._simulation_last_event_time = time.time() # Reset timer at start of listener

        while not self._stop_event.is_set():
            time.sleep(self.sim_listener_tick) 
            current_time = time.time()
            elapsed_since_last_event = current_time - self._simulation_last_event_time

            if not self._sim_objective_currently_found:
                if elapsed_since_last_event >= self.sim_objective_find_interval:
                    self._objective_found = True
                    self._sim_objective_currently_found = True
                    self._simulation_last_event_time = current_time
                    print(f"Simulated SSE Listener: Event 'objective_found_event'. Objective FOUND. (Cycle time: {elapsed_since_last_event:.2f}s)")
            else: # Objective is currently found
                if elapsed_since_last_event >= self.sim_objective_reset_interval:
                    self._objective_found = False
                    self._sim_objective_currently_found = False
                    self._simulation_last_event_time = current_time
                    print(f"Simulated SSE Listener: Event 'objective_reset_event'. Objective RESET. (Cycle time: {elapsed_since_last_event:.2f}s)")
            
        print("Simulated SSE Listener: Stopped.")

    def _live_listen_to_stream(self):
        """Listens to a real SSE stream."""
        if not SSEClient or not requests:
            print("Live SSE Listener Error: SSE client libraries not available.")
            return

        print(f"Live SSE Listener: Attempting to connect to stream at {self.stream_url}...")
        try:
            self.sse_client_instance = SSEClient(self.stream_url)
            for msg in self.sse_client_instance:
                if self._stop_event.is_set():
                    print("Live SSE Listener: Stop event received, exiting listener loop.")
                    break
                
                print(f"Live SSE Listener: Received message - Event: '{msg.event}', Data: '{msg.data}'")
                
                if msg.event == "objective_found_event":
                    print("Live SSE Listener: 'objective_found_event' received. Setting objective found to True.")
                    self._objective_found = True
                elif msg.event == "objective_reset_event":
                    print("Live SSE Listener: 'objective_reset_event' received. Setting objective found to False.")
                    self._objective_found = False

        except requests.exceptions.ConnectionError as e:
            print(f"Live SSE Listener: Connection error to {self.stream_url}: {e}.")
            self.sse_client_instance = None
        except Exception as e:
            print(f"Live SSE Listener: An unexpected error occurred: {e}")
        finally:
            if self.sse_client_instance and hasattr(self.sse_client_instance, 'close'):
                print(f"Live SSE Listener: Attempting to close SSEClient instance.")
                try:
                    self.sse_client_instance.close()
                    print("Live SSE Listener: SSEClient instance closed successfully.")
                except Exception as e_close:
                    print(f"Live SSE Listener: Error during SSEClient.close(): {e_close}")
            self.sse_client_instance = None
            print(f"Live SSE Listener: Thread for stream {self.stream_url} has finished.")

    def get_drone_data(self) -> SensorsData:
        if self.simulate:
            # Simulate slight GPS drift
            self._current_simulated_gps[0] += random.uniform(-self.sim_gps_drift_scale, self.sim_gps_drift_scale)
            self._current_simulated_gps[1] += random.uniform(-self.sim_gps_drift_scale, self.sim_gps_drift_scale)
            self._current_simulated_gps[2] += random.uniform(-self.sim_gps_drift_scale * 100, self.sim_gps_drift_scale * 100) # Altitude drift can be larger
            
            # Keep altitude somewhat bounded around its initial z component
            if not (self.sim_gps_center[2] - 5 < self._current_simulated_gps[2] < self.sim_gps_center[2] + 5):
                 self._current_simulated_gps[2] = self.sim_gps_center[2] + random.uniform(-0.5, 0.5) # Reset with small variation

            gps = torch.tensor(self._current_simulated_gps, dtype=torch.float32)
            
            accel = torch.tensor([random.uniform(self.sim_accel_range[0], self.sim_accel_range[1])], dtype=torch.float32)
            gyro = torch.tensor([random.uniform(self.sim_gyro_range[0], self.sim_gyro_range[1]),
                                 random.uniform(self.sim_gyro_range[0], self.sim_gyro_range[1]),
                                 random.uniform(self.sim_gyro_range[0], self.sim_gyro_range[1])
                                ], dtype=torch.float32)
            teta = torch.tensor([random.uniform(self.sim_teta_yaw_range[0], self.sim_teta_yaw_range[1]),
                                 random.uniform(self.sim_teta_pitch_range[0], self.sim_teta_pitch_range[1])
                                ], dtype=torch.float32)
            return SensorsData(gps=gps, accel=accel, gyro=gyro, teta=teta)
        else:
            # ... (live mode get_drone_data as before) ...
            print("API: get_drone_data() called in LIVE mode (implementation pending for real hardware). Returning dummy data.")
            return SensorsData(
                gps=torch.tensor([0.0,0.0,0.0], dtype=torch.float32), 
                accel=torch.tensor([0.0], dtype=torch.float32), 
                gyro=torch.tensor([0.0,0.0,0.0], dtype=torch.float32), 
                teta=torch.tensor([0.0,0.0], dtype=torch.float32)
            )

    def is_drone_find_objective(self) -> bool:
        return self._objective_found

    def stop_listening(self):
        """Stops the listener thread gracefully."""
        print(f"API: Signaling listener thread to stop (Simulate: {self.simulate})...")
        self._stop_event.set()
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=5)
            if self.listener_thread.is_alive():
                print("API Warning: Listener thread did not stop in time.")
            else:
                print("API: Listener thread stopped successfully.")
        else:
            print("API: Listener thread not running or already stopped.")

    # Ensure cleanup when the API object is deleted
    def __del__(self):
        print("API object is being deleted. Ensuring listener is stopped.")
        self.stop_listening()

# Example of how to run this file directly for testing the API simulation
if __name__ == '__main__':
    print("--- Testing API in Simulation Mode (Default Parameters) ---")
    sim_api_default = API(simulate=True)
    try:
        for i in range(5): 
            print(f"Main Test Loop (Default Sim {i+1}/5): Objective found? {sim_api_default.is_drone_find_objective()}")
            drone_data = sim_api_default.get_drone_data()
            if drone_data:
                 print(f"  -> Got Drone Data: GPS: {drone_data.gps.tolist()}")
            time.sleep(3) 
    finally:
        print("--- Main Test Loop: Stopping default simulated API ---")
        sim_api_default.stop_listening()
        print("--- Main Test Loop: Default sim test finished ---")

    print("--- Testing API in Simulation Mode (Custom Parameters) ---")
    custom_sim_api = API(
        simulate=True,
        sim_objective_find_interval_seconds=5.0, # Find faster
        sim_objective_reset_interval_seconds=7.0, # Reset faster
        sim_listener_tick_seconds=0.25,
        sim_initial_gps=[10.0, 20.0, 5.0],
        sim_gps_drift_scale=0.001,
        sim_accel_range=(-0.2, 0.2),
        sim_gyro_range=(-0.05, 0.05)
    )
    try:
        for i in range(10): 
            print(f"Main Test Loop (Custom Sim {i+1}/10): Objective found? {custom_sim_api.is_drone_find_objective()}")
            drone_data = custom_sim_api.get_drone_data()
            if drone_data:
                 print(f"  -> Got Drone Data: GPS: {drone_data.gps.tolist()}, Accel: {drone_data.accel.tolist()}")
            time.sleep(1) 
    except KeyboardInterrupt:
        print("\n--- Main Test Loop: Keyboard interrupt detected (Custom Sim) ---")
    finally:
        print("--- Main Test Loop: Stopping custom simulated API ---")
        custom_sim_api.stop_listening()
        print("--- Main Test Loop: Custom sim test finished ---")
    
    # ... (live mode test remains the same) ...
    print("\n\n--- Testing API in (Attempted) Live Mode ---")
    print("--- This will likely show connection errors if no server is at the default URL ---")
    live_api = API(stream_url="http://localhost:8000/events", simulate=False) 
    try:
        for i in range(3):
            print(f"Main Test Loop (Live {i+1}/3): Objective found? {live_api.is_drone_find_objective()}")
            live_data = live_api.get_drone_data() 
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n--- Main Test Loop: Keyboard interrupt detected (Live) ---")
    finally:
        print("--- Main Test Loop: Stopping live API ---")
        live_api.stop_listening()
        print("--- Main Test Loop: Live test finished ---")

