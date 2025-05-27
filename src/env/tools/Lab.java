package tools;

import java.io.IOException;
import java.util.*;
import java.util.logging.*;
import com.google.common.collect.Sets;
import ch.unisg.ics.interactions.wot.td.ThingDescription;
import ch.unisg.ics.interactions.wot.td.ThingDescription.TDFormat;
import ch.unisg.ics.interactions.wot.td.affordances.ActionAffordance;
import ch.unisg.ics.interactions.wot.td.affordances.Form;
import ch.unisg.ics.interactions.wot.td.affordances.PropertyAffordance;
import ch.unisg.ics.interactions.wot.td.clients.TDHttpRequest;
import ch.unisg.ics.interactions.wot.td.clients.TDHttpResponse;
import ch.unisg.ics.interactions.wot.td.io.TDGraphReader;
import ch.unisg.ics.interactions.wot.td.schemas.DataSchema;
import ch.unisg.ics.interactions.wot.td.schemas.BooleanSchema;
import ch.unisg.ics.interactions.wot.td.schemas.ObjectSchema;
import ch.unisg.ics.interactions.wot.td.vocabularies.TD;

/**
 * An class that extends {@link LearningEnvironment} for representing a learning
 * environment which can be used for Q learning.
 *
 * <p>
 * A Lab instance manages the state space and the action space of
 * environments that are similar to the lab of the Interactions group
 * (simulated or real environments), given the W3C Web of Things Thing
 * Description of the lab environment.
 * </p>
 */
public class Lab extends LearningEnvironment {

  /**
   * The W3C Web of Things Thing Description used for interacting with the
   * lab environment
   */
  private ThingDescription td;

  /**
   * The current state of the lab (and of every state of the lab) is formed as a
   * a list of 7 integer values: [z1Level, z2Level, z1Light, z2Light, z1Blinds,
   * z2Blinds, sunshine]:
   * <ul>
   * <li>z1Level: the level of light in Zone 1
   * <li>z2Level: the level of light in Zone 2
   * <li>z1Light: the status of the lights in Zone 1
   * <li>z2Light: the status of the lights in Zone 2
   * <li>z1Blinds: the status of the blinds in Zone 1
   * <li>z2Blinds: the status of the blinds in Zone 2
   * <li>sunshine: the level of sunshine out of the lab
   * </ul>
   */
  protected List<Integer> currentState = Arrays.asList(new Integer[7]);

  /**
   * The state of the lab depends on the values of
   * z1Level, z2Level, z1Light, z2Light, z1Blinds, z2Blinds, sunshine
   */

  /**
   * z1Level: the level of light in Zone 1
   * Possible values: 0,1,2,3
   * Respective keys: 0,1,2,3
   */
  private static final HashMap<Integer, Integer> z1Level = new HashMap<>();

  /**
   * z2Level: the level of light in Zone 2
   * Possible values: 0,1,2,3
   * Respective keys: 0,1,2,3
   */
  private static final HashMap<Integer, Integer> z2Level = new HashMap<>();

  /**
   * z1Light: the status of the lights in Zone 1
   * Possible values: false, true
   * Respective keys: 0,1
   */
  private static final HashMap<Integer, Boolean> z1Light = new HashMap<>();

  /**
   * z2Light: the status of the lights in Zone 2
   * Possible values: false, true
   * Respective keys: 0,1
   */
  private static final HashMap<Integer, Boolean> z2Light = new HashMap<>();

  /**
   * z1Blinds: the status of the blinds in Zone 1
   * Possible values: false, true
   * Respective keys: 0,1
   */
  private static final HashMap<Integer, Boolean> z1Blinds = new HashMap<>();

  /**
   * z2Blinds: the status of the blinds in Zone 2
   * Possible values: false, true
   * Respective keys: 0,1
   */
  private static final HashMap<Integer, Boolean> z2Blinds = new HashMap<>();

  /**
   * sunshine: the level of sunshine out of the lab
   * Possible values: 0,1,2,3
   * Respective keys: 0,1,2,3
   */
  private static final HashMap<Integer, Integer> sunshine = new HashMap<>();

  private static final Logger LOGGER = Logger.getLogger(Lab.class.getName());

  static {

    // possible substates for z1Level, z2Level, sunshine
    for (int i = 0; i < 4; i++) {
      z1Level.put(i, i);
      z2Level.put(i, i);
      sunshine.put(i, i);
    }

    // possible substates for z1Light
    z1Light.put(0, false);
    z1Light.put(1, true);

    // possible substates for z2Light
    z2Light.put(0, false);
    z2Light.put(1, true);

    // possible substates for z1Blinds
    z1Blinds.put(0, false);
    z1Blinds.put(1, true);

    // possible substates for z2Blinds
    z2Blinds.put(0, false);
    z2Blinds.put(1, true);

  };

  /**
   * A {@link Lab} instance is constructed based on the URL of the W3C Web of
   * Things
   * Thing Description of a lab (simulated or real)
   *
   * @param url The location of the W3C Web of Things Thing Description
   */
  public Lab(String url) {

    try {

      // Read the Thing Description from the URL
      this.td = TDGraphReader.readFromURL(TDFormat.RDF_TURTLE, url);

      // Create the state space
      createStateSpace();
      LOGGER.info("The lab has a state space of n=" + stateSpace.size());

      // Print the states of the state space
      LOGGER.info(String.valueOf(stateSpace));

      // Create the action space
      createActionSpace();
      LOGGER.info("The lab has an action space of m=" + actionSpace.size());

      // Print the actions of the action space
      for (Action action : actionSpace.values()) {
        LOGGER.info(action.toString());
      }

      // Read the current state
      readCurrentState();
      LOGGER.info("The lab current state: " + this.currentState);

    } catch (IOException e) {
      LOGGER.severe(e.getMessage());
    }
  }

  /**
   * @see {@link LearningEnvironment#getCompatibleStates(List)}
   */
  @Override
  public List<Integer> getCompatibleStates(List<Object> stateDescription) {

    List<Integer> compatibleStates = new ArrayList<>();
    List<List<Integer>> stateList = new ArrayList<>(stateSpace);

    for (int i = 0; i < stateList.size(); i++) {
      List<Integer> state = stateList.get(i);

      List<Object> substates = new ArrayList<>();

      substates.add(Lab.z1Level.get(state.get(0)));
      substates.add(Lab.z2Level.get(state.get(1)));
      substates.add(Lab.z1Light.get(state.get(2)));
      substates.add(Lab.z2Light.get(state.get(3)));
      substates.add(Lab.z1Blinds.get(state.get(4)));
      substates.add(Lab.z2Blinds.get(state.get(5)));
      substates.add(Lab.sunshine.get(state.get(6)));

      if (Collections.indexOfSubList(substates, stateDescription) != -1) {
        compatibleStates.add(i);
        System.out.println(state);
      }
      ;
    }
    return compatibleStates;
  }

  /**
   * @see {@link LearningEnvironment#readCurrentState()}
   */
  @Override
  public int readCurrentState() {
    Optional<PropertyAffordance> p = this.td.getFirstPropertyBySemanticType("https://example.org/was#Status");

    if (p.isPresent()) {

      Optional<Form> f = p.get().getFirstFormForOperationType(TD.readProperty);
      DataSchema ds = p.get().getDataSchema();

      if (f.isPresent()) {

        TDHttpRequest request = new TDHttpRequest(f.get(), TD.readProperty);

        try {
          TDHttpResponse response = request.execute();
          Map<String, Object> status = response.getPayloadAsObject((ObjectSchema) ds);

          int z1Level = discretizeLightLevel((Double) status.get("http://example.org/was#Z1Level"));
          int z2Level = discretizeLightLevel((Double) status.get("http://example.org/was#Z2Level"));

          boolean z1Light = (Boolean) status.get("http://example.org/was#Z1Light");
          boolean z2Light = (Boolean) status.get("http://example.org/was#Z2Light");

          boolean z1Blinds = (Boolean) status.get("http://example.org/was#Z1Blinds");
          boolean z2Blinds = (Boolean) status.get("http://example.org/was#Z2Blinds");

          int sunshine = discretizeSunshine((Double) status.get("http://example.org/was#Sunshine"));

          currentState.set(0, z1Level);
          currentState.set(1, z2Level);
          currentState.set(2, z1Light ? 1 : 0);
          currentState.set(3, z2Light ? 1 : 0);
          currentState.set(4, z1Blinds ? 1 : 0);
          currentState.set(5, z2Blinds ? 1 : 0);
          currentState.set(6, sunshine);

        } catch (IOException e) {
          LOGGER.severe(e.getMessage());
        }
      }
    }

    List<List<Integer>> stateList = new ArrayList<>(stateSpace);
    return stateList.indexOf(this.currentState);
  }

  /**
   * @see {@link LearningEnvironment#getApplicableActions(int)}
   */
  @Override
  public List<Integer> getApplicableActions(int state) {

    List<Integer> applicableActions = new ArrayList<>();
    List<List<Integer>> stateList = new ArrayList<>(stateSpace);
    List<Integer> st = stateList.get(state);

    for (int action : actionSpace.keySet()) {

      Action a = actionSpace.get(action);

      int stateAxis = a.getApplicableOnStateAxis();
      int stateValue = a.getApplicableOnStateValue();

      if (st.get(stateAxis) == stateValue) {
        applicableActions.add(action);
      }
    }

    return applicableActions;
  }

  /**
   * @see {@link LearningEnvironment#performAction(int)}
   */
  @Override
  public void performAction(int action) {
    Action a = actionSpace.get(action);

    try {
      a.getRequest().execute();
      LOGGER.info(a.getRequest().toString());
    } catch (IOException e) {
      LOGGER.severe(e.getMessage());
    }

  }

  /**
   * Creates the action space of the lab
   */
  private void createActionSpace() {

    this.affordanceTypes = Arrays.asList(
        "http://example.org/was#SetZ1Light",
        "http://example.org/was#SetZ2Light",
        "http://example.org/was#SetZ1Blinds",
        "http://example.org/was#SetZ2Blinds");

    for (String affType : affordanceTypes) {

      Optional<ActionAffordance> a = this.td.getFirstActionBySemanticType(affType);

      if (a.isPresent()) {

        Optional<Form> f = a.get().getFirstFormForOperationType(TD.invokeAction);
        Optional<DataSchema> ds = a.get().getInputSchema();

        if (f.isPresent() && ds.isPresent()) {

          Map<String, DataSchema> props = ((ObjectSchema) ds.get()).getProperties();
          Map<String, Object> payload = new HashMap<>();

          for (String propName : props.keySet()) {

            DataSchema propDs = props.get(propName);

            if (propDs instanceof BooleanSchema) {
              for (boolean propValue : Arrays.asList(false, true)) {
                payload.put(propName, propValue);
                TDHttpRequest request = new TDHttpRequest(f.get(), TD.invokeAction);
                request.setObjectPayload((ObjectSchema) ds.get(), payload);
                Action action = new Action(affType, new Object[] { propName }, new Object[] { propValue }, request);
                actionSpace.put(actionSpace.size(), action);
              }
            }
          }
        }
      }
    }
    setApplicableActions();
  }

  /**
   * Maps lux values to light levels:
   * lux < 50 -> level 0
   * lux in [50,100) -> level 1
   * lux in [100,300) -> level 2
   * lux >= 300 -> level 3
   */
  private int discretizeLightLevel(Double value) {
    if (value < 50) {
      return 0;
    } else if (value < 100) {
      return 1;
    } else if (value < 300) {
      return 2;
    }
    return 3;
  }

  /**
   * Maps lux values to light levels:
   * lux < 50 -> level 0
   * lux in [50,200) -> level 1
   * lux in [200,700) -> level 2
   * lux >= 700 -> level 3
   */
  private int discretizeSunshine(Double value) {
    if (value < 50) {
      return 0;
    } else if (value < 200) {
      return 1;
    } else if (value < 700) {
      return 2;
    }
    return 3;
  }

  /**
   * Creates the state space of the lab
   */
  private void createStateSpace() {
    this.stateSpace = Sets.cartesianProduct(
        Collections.unmodifiableSet(z1Level.keySet()),
        Collections.unmodifiableSet(z2Level.keySet()),
        Collections.unmodifiableSet(z1Light.keySet()),
        Collections.unmodifiableSet(z2Light.keySet()),
        Collections.unmodifiableSet(z1Blinds.keySet()),
        Collections.unmodifiableSet(z2Blinds.keySet()),
        Collections.unmodifiableSet(sunshine.keySet()));
  }

  /**
   * Returns the action that is applicable based on a given substate
   */
  private Action getApplicableAction(String stateAxis, Boolean stateValue) {
    return actionSpace.values()
        .stream().filter(v -> stateAxis.equals(v.getActionTag()) &&
            Arrays.asList(v.getPayload()).contains(stateValue))
        .findFirst().get();
  }

  /**
   * Set the applicable actions for each substate
   */
  private void setApplicableActions() {

    Action z1LightOnValidAction = getApplicableAction("http://example.org/was#SetZ1Light", true);
    Action z1LightOffValidAction = getApplicableAction("http://example.org/was#SetZ1Light", false);
    Action z2LightOnValidAction = getApplicableAction("http://example.org/was#SetZ2Light", true);
    Action z2LightOffValidAction = getApplicableAction("http://example.org/was#SetZ2Light", false);
    Action z1BlindsUpValidAction = getApplicableAction("http://example.org/was#SetZ1Blinds", true);
    Action z1BlindsDownValidAction = getApplicableAction("http://example.org/was#SetZ1Blinds", false);
    Action z2BlindsUpValidAction = getApplicableAction("http://example.org/was#SetZ2Blinds", true);
    Action z2BlindsDownValidAction = getApplicableAction("http://example.org/was#SetZ2Blinds", false);

    z1LightOnValidAction.setApplicableOn(2, 0);
    z1LightOffValidAction.setApplicableOn(2, 1);
    z2LightOnValidAction.setApplicableOn(3, 0);
    z2LightOffValidAction.setApplicableOn(3, 1);
    z1BlindsUpValidAction.setApplicableOn(4, 0);
    z1BlindsDownValidAction.setApplicableOn(4, 1);
    z2BlindsUpValidAction.setApplicableOn(5, 0);
    z2BlindsDownValidAction.setApplicableOn(5, 1);
  }

  /** Allow other classes to see the full list of 7-entry states */
  public List<List<Integer>> getStateSpace() {
    // return an unmodifiable view so nobody accidentally mutates it
    return this.stateSpace.stream()
        .map(Collections::unmodifiableList)
        .toList();
  }
}