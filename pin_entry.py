import multiprocessing as mp
import time

class PINManager:
    def __init__(self, authorized_count_queue):
        """
        Initialize PIN management system
        
        Args:
            authorized_count_queue (mp.Queue): Queue to share authorized count
        """
        self.authorized_count_queue = authorized_count_queue
    
    def start_pin_entry(self):
        """
        Continuously prompt for PIN entry and manage authorized count
        """
        while True:
            print("\n--- Security Access Control ---")
            pin = input("Enter 3-digit PIN (or 'q' to quit): ").strip()
            
            if pin.lower() == 'q':
                # Signal to exit
                self.authorized_count_queue.put(None)
                break
            
            # Hardcoded PIN for demo (can be replaced with more secure method)
            if pin == '123':
                # Put authorized count
                self.authorized_count_queue.put(1)
                print("✅ Authorized Successfully!")
            else:
                print("❌ Invalid PIN. Access Denied.")

def start_pin_manager(authorized_count_queue):
    """
    Function to start PIN manager in a separate process
    
    Args:
        authorized_count_queue (mp.Queue): Queue to share authorized count
    """
    pin_manager = PINManager(authorized_count_queue)
    pin_manager.start_pin_entry()

def main():
    # Create a queue for sharing authorized count
    authorized_count_queue = mp.Queue()
    
    # Create and start PIN manager process
    pin_process = mp.Process(target=start_pin_manager, args=(authorized_count_queue,))
    pin_process.start()
    
    # Keep main process running
    pin_process.join()

if __name__ == "__main__":
    main()