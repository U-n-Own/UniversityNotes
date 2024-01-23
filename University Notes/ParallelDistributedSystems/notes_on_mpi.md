# Just a quick reminder

- MPI low-level abroction.
- Allows prossses to communicate
- Rank is the logical name
- Total number of processes

2 PRIMITIVES:
- Send and Receive!
They are blocking
- BUFFER CAN BE REUSED
- YOU CAN CONTINUE WITH YOUR COMPUTATION
- COMM. DEVIKE
J
PORTABILITY
- EACH MPI program:
  1. STARTS WITH: MPI-INIT
  2. FINISHES with mpi finalize

- We have to compile with MPICC and MPIRUN or MPIEXEC
- we can specify the number of workers with the -n flag on MPIRUN

# Second MPI lecture
Basically, with MPI we have de ability to let processes communicate. By using this ability, we can for sure let one process P1 to send a message to P2 that wants to receive that message. Of course, we can do more than sending messages. We can say that: we have this communication, having P1 -> P2 but we wana P2 -> P3. This looks like a pipeline, right? P1 -> P2 -> P3. Once we have this pipeline, we can say: ok! what is preventing us to say that P2 is something more than a simple pipeline stage? We can consider the possibility to assign P2 to a FARM-like structure.


## Now, let's discuss this new "openmpi paradigm"
We mentioned the fact that our comp. is a matter of: initiating a process using `MPI_INIT()`. After that,we wanna understand what is our rank, aka who we are. Given the name of the communicator -> `MPI_COMM_WORLD` -> new variable with our rank (our id, unique name).
Then, we want to understand which processes belong to the world. This will made be possible with `MPI_COMM_SIZE()`
Then, if we wanna, let's say at least 5 procs to execute our code:

```C++
if(n_procs < 5){ exit(0); }
```

Now, what is the variable discriminating the things i wanna do?
The rank: for example, in the last lecture, we discussed how different ranks could be used: e.g. if my rank is 0 -> do something, else, do something else.

Messages don't only work to pass data, but most of the time they synchronize communication.
### First part: Emitter design
How to assign ranks inside a FARM? 0 Emitter, 1 Collector, all other ranks to workers!
What we could say: OK! let's start with 0 -> our emitter. We may want then to initiate a for loop that, for each message in the message to send, prepares the data to be sent, then `MPI_SEND(&task, task_size, MPI_INT, destination, TAG, MPI_COMM_WORLD);` Finally, increase destination to send another task to another worker! `destination = (destination + 1) % n_proc` We but have to exclude some workers, (emitter and collector) so -> `if (destination == 0){ destination = 2; }` Is our emitter complete? Almost.

Emitter: another for, this time not longer a for that starts from 0 to n_msgs. This time will start from 0 to number of processes. For each iteration, we send an EOS from the emitter once data are finished, so:``` MPI_SEND(&task, 1, MPI_INT, **EOS**, MPI_COMM_WORLD);```

### Second part: Collector design.
Remind: emitter ID is 1. so, in the previous switch that was checking our rank:

```C++
case 1:
    MPI_STATUS status;
    int end = FALSE;
    int eos = n_procs - 2
```

Let's suppose that the collector is working this way -> while loop waiting for EOS. `while(!eos) { ... }`
Inside the while loop: 
- `MPI_Recv(&res, 1, MPI_INT, MPI_ANY_SOURCE`, MPI_ANY_TAG, MPI_COMM_WORLD, &status);`
- Then, we may want to see from which tag did the received message come from.
- `if(status.MPI_TAG == TASK){ // process worker output } else {eos--; eos ? 0 : end=TRUE; break;}`

### Third part: Worker code
```C++
//...
default:
    int task;
    int end = FALSE
    MPI_Status status;
    // similarly for the collector
    while(!end){
	MPI_Recv(&task, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, &status);
	// note, for the source we could have had only the emitter rank!
	if(status.MPI_TAG == TASK){
	    // process input 
	    printf("work!!!");
	    task++;
	    MPI_Send(&task, 1, MPI_INT, TASK, MPI_COMM_WORLD); 
	}else {
	    MPI_Send(&task, 1, MPI_INT, 1, EOS, MPI_COMM_WORLD);
	    end = TRUE;
	}
        
    }
    MPI_Finalize();
    return;
// end of main
```

















  












