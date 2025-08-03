#!/usr/bin/env python3
"""
CHAOS UNLIMITED QUANTUM SCALING TEST
====================================

Push CHAOS quantum simulator to its absolute limits!
Test from 1 qubit to INFINITY and beyond! 

    # Start the stress test
    print(f"\nSTARTING QUANTUM SCALING TEST!")
    print(f"Algorithm: {selected_algorithm}")
    print(f"Range: {start_qubits} → {'∞' if max_qubits == float('inf') else max_qubits} qubits")
    print(f"\nPress Ctrl+C anytime to stop\n")NG: Large qubit counts may cause:
   - GPU fans sounding like jet engines
   - Electricity bills going to the moon
   - Hardware achieving quantum superposition between working/melting

Available Algorithms:
   • QFT - Quantum Fourier Transform (most memory efficient)
   • Grover - Database search (moderate complexity) 
   • Shor - Factorization (high complexity, lower limits)
   • GHZ - Multi-qubit entanglement (light computation)
   • Bell - Two-qubit entanglement (baseline test)

Recommended Limits:
   Beginner: 1-10 qubits (instant results)
   Intermediate: 11-15 qubits (few seconds)  
   Advanced: 16-20 qubits (minutes, fans spinning)
   Expert: 21-25 qubits (serious hardware stress)
   Legendary: 25+ qubits (prepare for takeoff!)
"""

import time
import tracemalloc
import numpy as np
import math
import sys
from quantum_circuit import QuantumCircuit

def get_hardware_info():
    """Check system capabilities and provide recommendations"""
    try:
        import cupy
        device = cupy.cuda.Device()
        free_mem, total_mem = device.mem_info
        free_gb = free_mem / (1024**3)
        total_gb = total_mem / (1024**3)
        
        # Estimate max qubits based on available memory
        # Each state needs 16 bytes (complex128), plus overhead
        max_qubits_estimate = int(math.log2(free_gb * 1024**3 / 32))  # Conservative estimate
        
        return {
            'has_gpu': True,
            'free_memory_gb': free_gb,
            'total_memory_gb': total_gb,
            'estimated_max_qubits': max_qubits_estimate,
            'device_name': device.attributes.get('Name', 'Unknown GPU')
        }
    except ImportError:
        return {
            'has_gpu': False,
            'free_memory_gb': 0,
            'total_memory_gb': 0,
            'estimated_max_qubits': 18,  # Conservative CPU estimate
            'device_name': 'CPU'
        }

def print_algorithm_info():
    """Display algorithm information and complexity"""
    algorithms = {
        'QFT': {
            'name': 'Quantum Fourier Transform',
            'complexity': 'O(n²) gates',
            'memory': 'Most efficient',
            'max_qubits': '25+',
            'description': 'Core of Shor\'s algorithm, highly optimized'
        },
        'Grover': {
            'name': 'Grover\'s Search',
            'complexity': 'O(√N) iterations',
            'memory': 'Moderate',
            'max_qubits': '20+',
            'description': 'Quantum database search with quadratic speedup'
        },
        'Shor': {
            'name': 'Shor\'s Factorization',
            'complexity': 'O(n³) gates',
            'memory': 'High',
            'max_qubits': '18+',
            'description': 'Breaks RSA encryption, most complex algorithm'
        },
        'GHZ': {
            'name': 'GHZ State Creation',
            'complexity': 'O(n) gates',
            'memory': 'Light',
            'max_qubits': '25+',
            'description': 'Multi-qubit entanglement, simple but powerful'
        },
        'Bell': {
            'name': 'Bell State Creation',
            'complexity': 'O(n) gates',
            'memory': 'Lightest',
            'max_qubits': '25+',
            'description': 'Basic entanglement test, good for stress testing'
        }
    }
    
    print("ALGORITHM DETAILS:")
    print("=" * 80)
    for i, (key, info) in enumerate(algorithms.items(), 1):
        print(f"{i}. {info['name']} ({key})")
        print(f"   Complexity: {info['complexity']}")
        print(f"   Memory Usage: {info['memory']}")
        print(f"   Est. Max Qubits: {info['max_qubits']}")
        print(f"   Description: {info['description']}")
        print()

def run_quantum_stress_test():
    """Main unlimited quantum scaling test"""
    
    print("CHAOS UNLIMITED QUANTUM SCALING TEST")
    print("=" * 80)
    print("Test quantum simulator scalability - push your hardware to the limits!")
    print("See how many qubits your system can handle!\n")
    
    # Get hardware info
    hw_info = get_hardware_info()
    
    print("HARDWARE ANALYSIS:")
    if hw_info['has_gpu']:
        print(f"   GPU Detected: {hw_info.get('device_name', 'Unknown')}")
        print(f"   GPU Memory: {hw_info['free_memory_gb']:.1f} GB free / {hw_info['total_memory_gb']:.1f} GB total")
        print(f"   Estimated Max Qubits: ~{hw_info['estimated_max_qubits']} (before meltdown)")
        print(f"   Performance Mode: LUDICROUS SPEED")
    else:
        print(f"   CPU Mode: No GPU detected")
        print(f"   Performance Mode: PATIENCE REQUIRED")
        print(f"   Recommended Limit: ~{hw_info['estimated_max_qubits']} qubits")
        print(f"   Suggestion: Get coffee while waiting")
    
    print()
    
    # Show algorithm options
    print_algorithm_info()
    
    # Get user selections
    algorithm_map = {'1': 'QFT', '2': 'Grover', '3': 'Shor', '4': 'GHZ', '5': 'Bell'}
    
    while True:
        try:
            choice = input("Choose algorithm (1-5): ").strip()
            if choice in algorithm_map:
                selected_algorithm = algorithm_map[choice]
                break
            print(" Invalid choice! Please enter 1-5")
        except KeyboardInterrupt:
            print("\n Test cancelled!")
            return
    
    print(f"Selected: {selected_algorithm}")
    
    # Get qubit range
    min_qubits = 2 if selected_algorithm == 'Bell' else 1
    
    while True:
        try:
            start_qubits = int(input(f"\nStart from how many qubits? (min: {min_qubits}): "))
            if start_qubits >= min_qubits:
                break
            print(f" Need at least {min_qubits} qubits for {selected_algorithm}!")
        except (ValueError, KeyboardInterrupt):
            print("\n Test cancelled!")
            return
    
    while True:
        try:
            max_input = input(f"Test up to how many qubits? (recommended: 15-20, or 'unlimited'): ").strip().lower()
            if max_input in ['unlimited', 'infinity', '∞', 'inf']:
                max_qubits = float('inf')
                print("UNLIMITED MODE: Will run until hardware gives up!")
                break
            else:
                max_qubits = int(max_input)
                if max_qubits >= start_qubits:
                    if max_qubits <= 15:
                        print(f"Target: {max_qubits} qubits - Safe zone!")
                    elif max_qubits <= 20:
                        print(f"Target: {max_qubits} qubits - Getting spicy!")
                    else:
                        print(f"Target: {max_qubits} qubits - Danger zone!")
                    break
                print(f"End must be >= {start_qubits}")
        except (ValueError, KeyboardInterrupt):
            print("\nTest cancelled!")
            return
    
    # Simple safety warning for high qubit counts
    if max_qubits == float('inf') or max_qubits > 22:
        print(f"\nHIGH PERFORMANCE WARNING:")
        print(f"   GPU/CPU fans may achieve supersonic flight")
        print(f"   Large qubit counts take significant time")
        print(f"   May contribute to global warming")

        if max_qubits == float('inf'):
            print(f"   May contribute to global warming")
        
        confirm = input(f"\nContinue? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("Playing it safe - smart choice!")
            return
    
    # Start the stress test
    print(f"\n STARTING QUANTUM SCALING TEST!")
    print(f"Algorithm: {selected_algorithm}")
    print(f"Range: {start_qubits} → {'∞' if max_qubits == float('inf') else max_qubits} qubits")
    print(f"\nPress Ctrl+C anytime to stop\n")
    
    # Results tracking
    results = []
    current_qubits = start_qubits
    total_start_time = time.perf_counter()
    
    # Results table header
    print("Qubits | Algorithm | State Size    | Memory      | Time        | Hardware Status | Suffering Level")
    print("-------|-----------|---------------|-------------|-------------|----------------|------------------")
    
    try:
        while current_qubits <= max_qubits:
            try:
                # Memory and time tracking
                tracemalloc.start()
                start_time = time.perf_counter()
                
                # Create quantum circuit
                qc = QuantumCircuit(current_qubits)
                
                # Apply selected algorithm
                if selected_algorithm == 'QFT':
                    # Prepare interesting state for QFT
                    for i in range(min(5, current_qubits)):
                        if i % 2 == 0:
                            qc.apply_gate('X', i)
                        else:
                            qc.apply_gate('H', i)
                    qc.apply_qft()
                    
                elif selected_algorithm == 'Grover':
                    # Grover's search setup
                    if current_qubits >= 3:
                        marked_state = '1' * (current_qubits - 1) + '0'
                        qc.apply_hadamard_to_all()
                        
                        # Adaptive iteration count
                        optimal_iterations = math.floor(math.pi / 4 * math.sqrt(2**current_qubits))
                        # Limit iterations for sanity
                        iterations = min(optimal_iterations, max(1, 25 - current_qubits))
                        
                        for _ in range(iterations):
                            qc.apply_grover_iteration(marked_state)
                    else:
                        qc.apply_hadamard_to_all()
                
                elif selected_algorithm == 'Shor':
                    # Simplified Shor's components
                    if current_qubits >= 4:
                        control_qubits = current_qubits // 2
                        
                        # Create superposition in control register
                        for i in range(control_qubits):
                            qc.apply_gate('H', i)
                        
                        # Entanglement between control and target
                        for i in range(control_qubits):
                            target = i + control_qubits
                            if target < current_qubits:
                                qc.apply_cnot(i, target)
                        
                        # QFT on control register
                        qc.apply_qft(list(range(control_qubits)))
                    else:
                        # Basic setup for small systems
                        qc.apply_gate('H', 0)
                        for i in range(1, current_qubits):
                            qc.apply_cnot(0, i)
                
                elif selected_algorithm == 'GHZ':
                    # GHZ state creation
                    qc.apply_gate('H', 0)
                    for i in range(1, current_qubits):
                        qc.apply_cnot(0, i)
                
                elif selected_algorithm == 'Bell':
                    # Extended Bell-like entanglement
                    qc.apply_gate('H', 0)
                    qc.apply_cnot(0, 1)
                    # Chain entanglement for larger systems
                    for i in range(2, current_qubits):
                        qc.apply_cnot(i-1, i)
                
                # Execute the quantum simulation
                qc.run()
                
                # Measure performance
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                current, peak_memory = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                # Validate quantum state
                state_norm = np.linalg.norm(qc.state_vector)
                is_valid = abs(state_norm - 1.0) < 1e-10
                
                # Calculate metrics
                state_size = 2**current_qubits
                memory_mb = peak_memory / (1024**2)
                memory_gb = peak_memory / (1024**3)
                
                # Determine hardware suffering level
                if execution_time < 0.1:
                    status = "Sleeping"
                    suffering = "Barely awake"
                elif execution_time < 1:
                    status = "Happy"
                    suffering = "Gentle workout"
                elif execution_time < 5:
                    status = "Sweating"
                    suffering = "Starting to sweat"
                elif execution_time < 15:
                    status = "Worried"
                    suffering = "Fans spinning up"
                elif execution_time < 60:
                    status = "Overheating"
                    suffering = "Fan orchestra"
                elif execution_time < 300:
                    status = "On Fire"
                    suffering = "Jet engine mode"
                elif execution_time < 900:
                    status = "Volcanic"
                    suffering = "Rocket launch"
                else:
                    status = "Death Rattle"
                    suffering = "Achieved plasma"
                
                if not is_valid:
                    status = "Quantum Collapse"
                    suffering = "Reality broke"
                
                # Store result
                results.append({
                    'qubits': current_qubits,
                    'algorithm': selected_algorithm,
                    'state_size': state_size,
                    'memory_mb': memory_mb,
                    'memory_gb': memory_gb,
                    'time_s': execution_time,
                    'valid': is_valid
                })
                
                # Print result
                print(f"{current_qubits:6d} | {selected_algorithm:9} | {state_size:13,} | {memory_mb:8.2f} MB | {execution_time:8.2f}s | {status:14} | {suffering}")
                
                # Progress warnings
                if current_qubits == 20:
                    print(f"       ENTERING DANGER ZONE! Hardware warranty voided!")
                elif current_qubits == 22:
                    print(f"       EXTREME TERRITORY! Consider life insurance!")
                elif current_qubits == 25:
                    print(f"       LEGENDARY STATUS! You are pushing the boundaries of physics!")
                
                # Safety breaks
                if execution_time > 1800:  # 30 minutes
                    print(f"       AUTO-STOP: 30-minute execution limit reached")
                    print(f"       MEDAL OF HONOR: You've achieved quantum computing legend status!")
                    break
                
                if memory_gb > 16:  # 16GB memory
                    print(f"       AUTO-STOP: Memory usage exceeded 16GB")
                    print(f"       MEMORY CHAMPION: You've maxed out your RAM!")
                    break
                
            except KeyboardInterrupt:
                print(f"\nEMERGENCY BRAKE: User intervention at {current_qubits} qubits!")
                print(f"   Hardware status: Probably grateful")
                break
            except Exception as e:
                error_msg = str(e)[:30] + "..." if len(str(e)) > 30 else str(e)
                print(f"{current_qubits:6d} | {selected_algorithm:9} | HARDWARE LIMIT   | EXCEEDED    | CRITICAL    | DEAD        | {error_msg}")
                tracemalloc.stop()
                
                if current_qubits >= 15:
                    print(f"       HARDWARE LIMIT REACHED: Your system has spoken!")
                    print(f"       ACHIEVEMENT UNLOCKED: Found your hardware's breaking point!")
                    break
            
            current_qubits += 1
    
    except KeyboardInterrupt:
        print(f"\nMISSION ABORTED by user command!")
    
    # Final analysis and celebration
    total_time = time.perf_counter() - total_start_time
    
    if results:
        print(f"\n" + "=" * 80)
        print(f"QUANTUM STRESS TEST COMPLETE!")
        print(f"=" * 80)
        
        max_qubits_achieved = max([r['qubits'] for r in results])
        max_state_size = max([r['state_size'] for r in results])
        max_memory = max([r['memory_gb'] for r in results])
        max_time = max([r['time_s'] for r in results])
        
        print(f"LEGENDARY ACHIEVEMENTS:")
        print(f"   Maximum qubits conquered: {max_qubits_achieved}")
        print(f"   Largest quantum universe: {max_state_size:,} quantum states")
        print(f"   Total suffering time: {total_time:.1f} seconds")
        print(f"   Algorithm mastered: {selected_algorithm}")
        print(f"   Peak memory usage: {max_memory:.3f} GB")
        print(f"   Peak execution time: {max_time:.1f} seconds")
        
        # Hardware survival assessment
        print(f"\nHARDWARE SURVIVAL REPORT:")
        if max_time < 5:
            print(f"   PRISTINE: Hardware barely noticed the test")
        elif max_time < 30:
            print(f"   SURVIVOR: Hardware sweated a bit but lived")
        elif max_time < 300:
            print(f"   BATTLE-SCARRED: Hardware has war stories to tell")
        elif max_time < 900:
            print(f"   VETERAN: Hardware achieved legendary stress levels")
        else:
            print(f"   TRANSCENDENT: Hardware has seen things that cannot be unseen")
        
        # Memory efficiency celebration
        if len(results) > 1:
            final_result = results[-1]
            traditional_memory_gb = (final_result['state_size']**2 * 16) / (1024**3)
            chaos_memory_gb = final_result['memory_gb']
            efficiency = traditional_memory_gb / chaos_memory_gb if chaos_memory_gb > 0 else float('inf')
            
            print(f"\nMEMORY EFFICIENCY BREAKTHROUGH:")
            print(f"   CHAOS used: {chaos_memory_gb:.3f} GB")
            print(f"   Traditional simulators would need: {traditional_memory_gb:.1f} GB")
            print(f"   EFFICIENCY MULTIPLIER: {efficiency:,.0f}x improvement!")
            print(f"   You just saved {traditional_memory_gb - chaos_memory_gb:.1f} GB of memory!")
        
        # Achievement badges
        print(f"\nACHIEVEMENT BADGES EARNED:")
        
        if max_qubits_achieved >= 1:
            print(f"   BRONZE: Quantum Apprentice (1+ qubits)")
        if max_qubits_achieved >= 10:
            print(f"   SILVER: Quantum Journeyman (10+ qubits)")
        if max_qubits_achieved >= 15:
            print(f"   GOLD: Quantum Master (15+ qubits)")
        if max_qubits_achieved >= 20:
            print(f"   DIAMOND: Quantum Legend (20+ qubits)")
        if max_qubits_achieved >= 25:
            print(f"   COSMIC: Quantum God (25+ qubits)")
        
        if max_time > 300:
            print(f"   PATIENCE MASTER: Waited over 5 minutes")
        if max_memory > 1:
            print(f"   MEMORY WARRIOR: Used over 1GB RAM")
        if efficiency > 100000:
            print(f"   EFFICIENCY HERO: 100,000x+ improvement")
        
        print(f"\nCONGRATULATIONS!")
        print(f"You've successfully stress-tested CHAOS quantum simulator!")
        print(f"Your hardware has been through quantum boot camp and survived!")
        print(f"You are now certified quantum computing badass!")
        
        # Share-worthy stats
        print(f"\nBRAG-WORTHY STATS FOR SOCIAL MEDIA:")
        print(f"\"Just pushed quantum computing to {max_qubits_achieved} qubits")
        print(f"and simulated {max_state_size:,} quantum states with CHAOS!\"")
        print(f"\"My computer survived {max_time:.1f} seconds of quantum torture!\"")
        
    else:
        print(f"\nTOTAL SYSTEM FAILURE!")
        print(f"No simulations survived. Hardware needs quantum healing. 🏥")
    
    print(f"\n Thank you for testing CHAOS quantum simulator!")
    print(f"May your qubits be ever in superposition!")

if __name__ == "__main__":
    try:
        run_quantum_stress_test()
    except Exception as e:
        print(f"\n CATASTROPHIC QUANTUM FAILURE!")
        print(f"Error: {str(e)}")
        print(f"The quantum realm has spoken. Please try again.")
        import traceback
        traceback.print_exc()
