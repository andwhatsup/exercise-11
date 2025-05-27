//illuminance controller agent

/*
* The URL of the W3C Web of Things Thing Description (WoT TD) of a lab environment
* Simulated lab WoT TD: "https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl"
* Real lab WoT TD: Get in touch with us by email to acquire access to it!
*/

/* Initial beliefs and rules */

// the agent has a belief about the location of the W3C Web of Thing (WoT) Thing Description (TD)
// that describes a lab environment to be learnt
learning_lab_environment("https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl").
real_lab_environment("https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab-real.ttl").
// the agent believes that the task that takes place in the 1st workstation requires an indoor illuminance
// level of Rank 2, and the task that takes place in the 2nd workstation requires an indoor illumincance 
// level of Rank 3. Modify the belief so that the agent can learn to handle different goals.
task_requirements([2,3]).

/* Initial goals */
!start. // the agent has the goal to start

/* 
 * Plan for reacting to the addition of the goal !start
 * Triggering event: addition of goal !start
 * Context: the agent believes that there is a WoT TD of a lab environment located at Url, and that 
 * the tasks taking place in the workstations require indoor illuminance levels of Rank Z1Level and Z2Level
 * respectively
 * Body: (currently) creates a QLearnerArtifact and a ThingArtifact for learning and acting on the lab environment.
*/
@start
+!start : learning_lab_environment(Url) & real_lab_environment(RealLab)
  & task_requirements([Z1Level, Z2Level]) <-

  .print("Hello world");


  // creates a QLearner artifact for learning the lab Thing described by the W3C WoT TD located at URL
  makeArtifact("qlearner", "tools.QLearner", [Url], QLArtId);

  calculateQ([Z1Level, Z2Level], 100, 0.5, 0.8, 0.4, 15);

  // creates a ThingArtifact artifact for reading and acting on the state of the lab Thing
  makeArtifact("lab", "org.hyperagents.jacamo.artifacts.wot.ThingArtifact", [Url], LabArtId);
  !achieve_goal.


// Plan: achieve_goal
// Purpose: Executes actions to reach desired illuminance levels in two workstations
+!achieve_goal : task_requirements([Z1Level, Z2Level]) & not retry_count(_) <-
  +retry_count(0);
  !achieve_goal.

+!achieve_goal : task_requirements([Z1Level, Z2Level]) & retry_count(Count) & Count < 10 <-
  // 1. Print the target illuminance levels for both workstations
  .print("Trial ", Count + 1, " of 10");
  .print("Target illuminance - Workstation 1: ", Z1Level, ", Workstation 2: ", Z2Level);

  // 2. Read current state of the environment
  readProperty("https://example.org/was#Status", CurrentTags, CurrentState);
  .print("Environment status - Tags: ", CurrentTags, ", State: ", CurrentState);

  // 3. Determine next action using Q-Learning algorithm
  getActionFromState([Z1Level,Z2Level], CurrentTags, CurrentState,
                               NextTag, NextPayloadTags, NextPayload);
  .print("Selected action - Type: ", NextTag, ", Tags: ", NextPayloadTags, ", Value: ", NextPayload);
  
  // 4. Execute the chosen action in the environment
  invokeAction(NextTag, NextPayloadTags, NextPayload);

  // 5. Wait for action to take effect
  .wait(5000);

  // 6. Check if goal state has been reached
  readProperty("https://example.org/was#Status", NewTags, NewState);
  isGoalReached([Z1Level,Z2Level], NewTags, NewState, Reached);
  if (Reached) { .print("Goal status: Achieved") } else { .print("Goal status: Not achieved yet") };

  // 7. Either terminate or continue with retry limit
  if (Reached == true) {
    .print("=== Success ===");
    .print("Target illuminance levels achieved: [", Z1Level, ",", Z2Level, "]");
    .print("Learning process completed");
  } else {
    -+retry_count(Count + 1);
    !achieve_goal;
  }.

+!achieve_goal : retry_count(Count) & Count >= 10 <-
  .print("=== Failed ===");
  .print("Maximum retries reached without achieving goal");
  .print("Learning process terminated").