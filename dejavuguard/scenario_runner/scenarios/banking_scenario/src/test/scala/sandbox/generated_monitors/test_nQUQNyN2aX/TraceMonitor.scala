
/* Generic Monitoring Code common for all properties. */

import net.sf.javabdd.{BDD, BDDFactory}
import java.io.*

import org.apache.commons.csv.{CSVFormat, CSVRecord}

import scala.collection.mutable.ListBuffer
import scala.jdk.CollectionConverters.*
import scala.util.{Try, Success, Failure}
import java.security.MessageDigest

object Options {
  var DEBUG: Boolean = false
  var PROFILE: Boolean = false
  var PRE_PREDICTION: Boolean = true
  var BITS: Int = 20
  var UNIT_TEST: Boolean = false
  var STATISTICS: Boolean = true
  var RESULT_FILE: String = ""
  var PRINTS_STAT: Boolean = true
}

object Util {
  type Binding = Map[String, Any]
  val emptyBinding: Binding = Map()

  var resultFile: PrintWriter = null
  var profileFile: BufferedWriter = null

  def openResultFile(name: String): Unit = {
    resultFile = new PrintWriter(new File(name))
  }

  def writelnResult(x: Any) = {
    resultFile.write(x.toString + "\n")
  }

  def closeResultFile(): Unit = {
    resultFile.close()
  }

  def openProfileFile(name: String): Unit = {
    val file = new File(name)
    profileFile = new BufferedWriter(new FileWriter(file))
  }

  def writeProfile(x: Any): Unit = {
    profileFile.write(x.toString)
  }

  def writelnProfile(x: Any): Unit = {
    profileFile.write(x.toString + "\n")
  }

  def writelnProfile(): Unit = {
    profileFile.write("\n")
  }

  def closeProfileFile(): Unit = {
    profileFile.close()
  }

  def debug(str: => String): Unit = {
    if (Options.DEBUG) println(str)
  }

  def bddToString(bdd: BDD): String = {
    if (bdd.isZero)
      "False"
    else if (bdd.isOne)
      "True"
    else
      bdd.toString
  }

  implicit class LiftBDD(bdd: BDD) {
    def dot(msg: String = "DEBUGGING"): Unit = {
      if (Options.DEBUG) {
        println("@@@@@@@@@@@@@@@@@@@@@")
        println(msg)
        if (bdd.isZero)
          println("False")
        else if (bdd.isOne)
          println("True")
        else
          bdd.printDot()
      }
    }
  }
}

import Util.*

/**
  * Patterns for checking whether the state contains a certain event.
  */

trait Pattern

case class V(name: String) extends Pattern {
  override def toString: String = name
}

case class C(value: Any) extends Pattern {
  override def toString: String = value.toString
}

/**
  * A state in a trace. A trace holds one event. Event patterns can be checked against
  * the state using the <code>holds</code> method.
  */

class State {
  type Event = (String, List[Any])

  var events: List[Event] = Nil
  var current: Event = null

  /**
    * Update for single-event state (backward compatible).
    *
    * @param name the name of the event.
    * @param args the arguments of the event.
    */

  def update(name: String, args: List[Any]): Unit = {
    current = (name, args)
    events = List(current)
  }

  /**
    * Add an event to the current composite state.
    */
  def addEvent(name: String, args: List[Any]): Unit = {
    val event = (name, args)
    events = events :+ event
    current = event
  }

  /**
    * Set the current event pointer (for iterating during evaluation).
    */
  def setCurrent(event: Event): Unit = {
    current = event
  }

  /**
    * Clear events for the next state.
    */
  def nextState(): Unit = {
    events = Nil
    current = null
  }

  /**
    * Matches an event pattern as it occurs in a formula against the current event.
    *
    * @param name     the name of the event.
    * @param patterns the argument patterns.
    * @return optional BDD in case there is a match. The BDD will represent the binding of
    *         variables to values.
    */

  def holds(name: String, patterns: List[Pattern]): Option[Binding] = {
    val (cname, cargs) = current
    if (cname != name) None else {
      assert(patterns.size == cargs.size,
        s"patterns '${patterns.mkString(",")}' do not match args: '${cargs.mkString(",")}'")
      var binding: Binding = emptyBinding
      var matched = true
      for ((pat, value) <- patterns.zip(cargs) if matched) {
        pat match {
          case C(v) =>
            if (v != value) matched = false
          case V(x) =>
            if (binding.isDefinedAt(x)) {
              if (binding(x) != value) matched = false
            } else {
              binding += (x -> value)
            }
        }
      }
      if (matched) Some(binding) else None
    }
  }

  /**
    * Match across all events in the composite state.
    */
  def holdsAny(name: String, patterns: List[Pattern]): List[Binding] = {
    events.flatMap { event =>
      val saved = current
      setCurrent(event)
      val result = holds(name, patterns)
      current = saved
      result.toList
    }
  }

  override def toString: String = {
    var result = ""
    result += "#########################################################\n"
    for (event <- events) {
      result += s"#### ${event._1}("
      result += event._2.mkString(",") + ")\n"
    }
    result += "#########################################################\n"
    result
  }
}


/**
 * Trait defining the behavior for pre-monitoring functionalities.
 *
 * PreMonitorTrait encapsulates the method(s) required to monitor
 * specific events and evaluate them based on various parameters.
 */
trait PreMonitorTrait {
  def evaluate(event_name: String, params: Any*): Option[Any]
}


/**
  * A variable is represented by an object of this class.
  *
  * @param F        the formula that the variable is part of.
  * @param name     the name of the variable, used for error messages
  * @param offset   the offset in the total bitvector where the bits for this variable start.
  * @param nrOfBits the number of bits allocated to represent values of this variable.
  */


class Variable(F: Formula)(name: String, bounded: Boolean, offset: Int, nrOfBits: Int) {
  val G = F.bddGenerator
  var bits: Array[Int] = (for (x <- offset + nrOfBits - 1 to offset by -1) yield x).toArray
  val quantvar: BDD = G.getQuantVars(bits)
  // needed to perform quantification.
  var next: Int = -1
  var bdds: Map[Any, BDD] = Map()
  val MAX = Math.pow(2, nrOfBits)
  val allOnes: BDD = {
    var result: BDD = G.True
    for (pos <- bits) {
      result = result.and(G.theOneBDDFor(pos))
    }
    result
  }
  val freeInitially: BDD = allOnes.not
  var free: BDD = freeInitially
  var seen: BDD = G.False
  var inRelation: BDD = G.False

  /**
    * Records the fact that a BDD for this variable occurs in a relation, thus
    * preventing it from being garbage collected.
    *
    * @param bdd the BDD being recorded as being part of a relation.
    */

  def inRelation(bdd: BDD): Unit = {
    if (!bounded) inRelation = inRelation.or(bdd) // only add if not already added (i.e. the variable is bounded)
  }

  /**
    * Returns the BDD corresponding to the value, according to the enumeration of the value.
    * Either it exists already or it is built.
    *
    * @param v the value for which a BDD must be created.
    * @return the BDD corresponding to <code>v</code>.
    */

  def getBddOf(v: Any): BDD = {
    if (bdds.contains(v)) {
      val result = bdds(v)
      result.dot(s"Looking up existing BDD for $v")
      result
    } else {
      free.dot(s"free before get new BDD for variable $name for positions ${bits.mkString(",")}")
      if (timeToGarbageCollect) collectGarbage()
      if (free.isZero) {

        writelnResult(s"${F.monitor.lineNr} oom")
        assert(false, s"Out of memory for variable $name!")
      }
      val result = free.satOne(allOnes, true)

      free = free.and(result.not())
      bdds += (v -> result)
      F.addTouchedByLastEvent(name, v, result)
      result.dot(s"BDD for $name=$v")
      if (bounded) {
        seen = seen.or(result)
        seen.dot("seen thereafter")
      }
      free.dot("free thereafter")
      result
    }
  }

  /**
    * Determines whether it is time to garbage collect for a variable.
    *
    * @return true if it is time to garbage collect.
    */

  def timeToGarbageCollect: Boolean = {
    !bounded && free.isZero
  }

  /**
    * Collects garbage for a variable.
    */

  def collectGarbage(): Unit = {
    F.monitor.garbageWasCollected = true
    debug("+++++ START GARBAGE COLLECTION +++++")
    free = freeInitially
    free.dot(s"##### free initially")
    for (i <- F.indices) {
      val bdd_i = G.getFreeBDDOf(name, F.pre(i)) // not sure we access now at the right time
      free = free.and(bdd_i)
      bdd_i.dot(s"##### bdd_i for index $i")
      free.dot(s"##### free this cycle around")
    }
    free.dot(s"++++++++++ free after garbage collection before taking uncollectable into account ++++++++++")
    inRelation.dot(s"++++++++++ uncollectable before garbage collection ++++++++++")
    free = free.and(inRelation.not())
    free.dot(s"++++++++++ free after garbage collection ++++++++++")
    removeGarbageValues()
  }

  /**
    * Called after the <code>collectGarbage()</code> has been called to remove all
    * value-BDD mappings, where the BDD has been garbage collected.
    */

  def removeGarbageValues(): Unit = {
    val values = bdds.keySet
    for (v <- values) {
      val bdd = bdds(v)
      if (bdd.imp(free).isOne) {
        debug(s"removing variable $name's entry for value $v")

        writelnResult(s"${F.monitor.lineNr} -- $v")
        bdds -= v
      }
    }
    debug(s"Remaining entries for variable $name: ${bdds.keySet.mkString(", ")}")
  }
}

/**
  * An object of this class represents all the variables in a formula,
  * including variables containing time values if timed temporal properties
  * occur in the property.
  *
  * It contains a mapping from variable names (strings) to objects of
  * class <code>Variable</code>, each of which contains the hashmap
  * from values of the corresponding variable to BDDs.
  *
  * @param variables      the variables in the formula, each indicated by
  *                       name, whether it is bounded (true = yes), and number of bits representing it.
  * @param bitsPerTimeVar the number of bits to be allocated per time variable.
  *                       This number is `0` if the property does not contain
  *                       timed temporal operators.
  */

class BDDGenerator(F: Formula)(variables: List[(String, Boolean, Int)], bitsPerTimeVar: Int) {
  var B: BDDFactory = BDDFactory.init(10000, 10000)
  val True: BDD = B.one()
  val False: BDD = B.zero()
  var offset: Int = 0
  val totalNumberOfBits: Int = variables.map(_._3).sum
  var varMap: Map[String, Variable] = Map()
  lazy val otherQuantVars: Map[String, List[BDD]] = {
    val varNames = variables.map(_._1)
    var result: Map[String, List[BDD]] = (for (varName <- varNames) yield (varName -> Nil)).toMap
    for (varName1 <- varNames; varName2 <- varNames if varName1 != varName2) {
      val otherQuantVarsSoFar = result(varName1)
      val newOtherQuant = varMap(varName2).quantvar
      result += (varName1 -> (newOtherQuant :: otherQuantVarsSoFar))
    }
    result
  }

  val nrOfTimeVariables = 5

  if (totalNumberOfBits > 0 || bitsPerTimeVar > 0) {
    B.setVarNum(totalNumberOfBits + (nrOfTimeVariables * bitsPerTimeVar))
  }

  /**
    * Returns a BDD for the bit positions provided as argument. The BDD is used to
    * represent the bits to quantify over for a particular DejaVu formula variable.
    *
    * @param bits the bit positions (variables) to include in the BDD.
    * @return a BDD over those variables.
    */

  def getQuantVars(bits: Array[Int]): BDD = {
    B.buildCube(0, bits).support()
  }

  /**
    * The BDD for a single position that is true only of that bit is 1.
    *
    * @param pos the position making part of the resulting BDD.
    * @return the BDD accepting on 1 for that position.
    */

  def theOneBDDFor(pos: Int): BDD = {
    B.ithVar(pos)
  }

  /**
    * Initializes the <code>varMap</code> variable by mapping each variable in the formula to
    * an instance of the <code>Variable</code> class.
    */

  def initializeVariables(): Unit = {
    for ((x, b, v) <- variables) {
      varMap += (x -> new Variable(F)(x, b, offset, v))
      offset += v
    }
  }

  /**
    * Get the BDD of value <code>v</code> when assigned to variable <code>x</code>.
    *
    * @param x the variable the value <code>v</code> is assigned to.
    * @param v the value being assigned to <code>x</code>.
    * @return the BDD representing the value <code>v</code>.
    */

  def getBddOf(x: String, v: Any): BDD =
    varMap(x).getBddOf(v)

  /**
    * Collects the garbage for a variable in a sub-formula. This is done using the formula:
    *
    * <code>
    * forall y0,...,z0,... . (F[1/x0,...,1/xn] <-> F)
    * </code>
    *
    * where <code>x</code> is the variable, and <code>x0,x1,...,xn</code> are the bit positions for that variable,
    * and <code>y0,...,z0,...</code> are the bit positions for all other variables <code>y, z, ...</code>.
    * The formula defines a BDD which accepts values <code>v</code> for <code>x</code> (in <code>F</code>)
    * such that <code>F[v/x]</code> is identical to <code>F[1/x0,...,1/xn]</code>. Those are the values
    * that are no longer needed, hence can be garbage collected. Recall that 111..1 represents all values not
    * yet seen.
    *
    * @param varName the name of the variable being garbage collected (<code>x</code> in the above example).
    * @param formula the formula being garbage collected over (<code>F</code> in the above example).
    * @return the free assignments.
    */

  def getFreeBDDOf(varName: String, formula: BDD): BDD = {
    val variable = varMap(varName)
    val formulaWithOnes = formula.restrict(variable.allOnes)
    var result = formulaWithOnes.biimp(formula)
    for (quantVar <- otherQuantVars(varName)) result = result.forAll(quantVar)
    result
  }
}

/**
  * Maintains trace statistics for a monitoring session. It specifically keeps track of
  * which events occur in the trace, how many times, and how this relates to the events
  * referred to in the specification. Can be useful for debugging a specification.
  *
  * @param events the events referred to in the specification
  */

class TraceStatistics(events: Set[String]) {
  var eventTable: Map[String, Long] = events.map(_ -> 0.asInstanceOf[Long]).toMap

  def update(eventName: String): Unit = {
    eventTable.get(eventName) match {
      case None => eventTable += (eventName -> 1)
      case Some(count) => eventTable += (eventName -> (count + 1))
    }
  }

  override def toString: String = {
    var result: String = ""
    result += "==================\n"
    result += "  Event Counts:\n"
    result += "------------------\n"
    val maxNameSize = eventTable.keySet.map(_.size).max
    for ((name, count) <- eventTable) {
      val spaces = maxNameSize - name.size
      val namePadded = name + (" " * spaces)
      result += f"  $namePadded : $count"
      if (count == 0) {
        result += " event did not occur in trace\n"
      } else if (!(events contains name)) {
        result += " unknown\n"
      } else {
        result += "\n"
      }
    }
    result += "==================\n"
    result
  }
}

/**
  * The generic Monitor class.
  * A specialized monitor for a set of properties must extend this class.
  * It contains the BDD generator (which generates the association between values
  * and BDDs), the state (which contains the current event), and the list of user
  * provided formulas. In addition it provides a set of options that can be set
  * by the user.
  */

abstract class Monitor(preMonitor: PreMonitorTrait) {
  val state: State = new State
  var formulae: List[Formula] = Nil
  var lineNr: Int = 0
  var garbageWasCollected: Boolean = false
  var statistics: TraceStatistics = new TraceStatistics(eventsInSpec)
  var currentTime: Int = 0
  var deltaTime: Int = 0
  var errors: Int = 0

  /**
    * Sets the current time to the time indicated by the timestamp associated
    * with the latest event. It specifically sets `deltaTime` to denote the
    * difference between the previous time stamp and this one.
    *
    * @param timeStamp the new time value for the latest event.
    */

  def setTime(timeStamp: Int): Unit = {
    deltaTime = timeStamp - currentTime
    currentTime = timeStamp
  }

  def preMonitor_(event_name: String, params: Any*): Option[Any] = {
    preMonitor.evaluate(event_name, params: _*)
  }

  /**
    * Returns the set of events referred to in the specification, either defined, or referred to
    * in the LTL formulas. Must be overridden by generated specification specific monitor.
    *
    * @return the set of referred to events.
    */

  def eventsInSpec: Set[String]

  /**
    * Used for timing performance. The timing is printed on standard output.
    *
    * @param block the code block that is being timed.
    * @tparam R the result type of the block.
    * @return the result of the block.
    */

  def time[R](block: => R): R = {
    val t1 = System.currentTimeMillis()
    val result = block
    val t2 = System.currentTimeMillis()
    val ms = (t2 - t1).toFloat
    val sec = ms / 1000
    println()
    println("Elapsed analysis time: " + sec + "s")
    result
  }

  /**
    * Submits an event to the monitor. This again causes the monitor evaluation to be
    * performed, which will evaluate all asserted formulas on this new event.
    *
    * @param name the name of the event.
    * @param args the arguments to the event.
    */

  def submit(name: String, args: List[Any]): Map[String, Boolean] = {
    if (Options.STATISTICS) {
      statistics.update(name)
    }
    state.update(name, args)
    evaluate()
  }

  /**
    * Submit a composite state with multiple events.
    * All events are evaluated together as one logical time step.
    */
  def submitState(events: List[(String, List[Any])]): Map[String, Boolean] = {
    state.nextState()
    for ((name, args) <- events) {
      state.addEvent(name, args)
      if (Options.STATISTICS) {
        statistics.update(name)
      }
    }
    evaluate()
  }

  /**
    * Vararg (variable length argument list) variant of method above. This form allows calls
    * like <code>submit("send",1,2)</code> rather than writing <code>submit("send",List(1,2))</code>.
    * Submits an event to the monitor. This again causes the monitor evaluation to be
    * performed, which will evaluate all asserted formulas on this new event.
    *
    * @param name the name of the event.
    * @param args the arguments to the event.
    */

  def submit(name: String, args: Any*): Unit = {
    submit(name, args.toList)
  }

  /**
    * Submits an entire trace to the monitor, as an alternative to submitting
    * events one by one. This method can only be called in offline monitoring.
    *
    * @param events the trace.
    */

  def submitTrace(events: List[(String, List[Any])]): Unit = {
    for ((event, args) <- events) {
      submit(event, args)
    }
    end()
  }

  /**
    * Submits an entire trace stored in CSV (Comma Separated Value format) format
    * to the monitor, as an alternative to submitting events one by one. This method
    * can only be called in offline monitoring.
    *
    * @param file the log file in CSV format to be verified.
    */

  def submitCSVFile(file: String): Unit = {
    val source = scala.io.Source.fromFile(file)
    val allLines = source.getLines().toList
    source.close()

    time {
      lineNr = 0
      if (allLines.isEmpty) {
        // empty file, nothing to process
      } else if (allLines.head.trim.startsWith("#format:")) {
        val format = allLines.head.trim.stripPrefix("#format:").trim.toLowerCase
        val dataLines = allLines.tail
        format match {
          case "normal" =>
            submitPlainCSVFromLines(dataLines, timed = false)
          case "timed" =>
            submitTimedCSVFromLines(dataLines)
          case "grouped" =>
            submitGroupedCSVFromLines(dataLines)
          case other =>
            println(s"*** Unknown format: $other. Treating as normal.")
            submitPlainCSVFromLines(dataLines, timed = false)
        }
      } else {
        // No header: use filename convention for backward compatibility.
        // Files with ".timed." in the name treat the last column as a timestamp.
        val timed = file.contains(".timed.")
        submitPlainCSVFromLines(allLines, timed)
      }

      println(s"Processed $lineNr events")
      end()
    }
  }

  private def submitPlainCSVFromLines(lines: List[String], timed: Boolean = false): Unit = {
    for (line <- lines) {
      lineNr += 1
      if (line.trim.nonEmpty) {
        printProgress()
        val in: Reader = new java.io.StringReader(line)
        val record = CSVFormat.DEFAULT.parse(in).asScala.head
        val name = record.get(0)
        var args = new ListBuffer[Any]()
        val eventSize = if (timed) {
          val ts = record.get(record.size() - 1).toInt
          setTime(ts)
          record.size() - 1
        } else {
          record.size()
        }
        if (eventSize > 1 && record.get(1).startsWith("[")) {
          args += (1 until eventSize).map(record.get).toList.mkString(", ")
        } else {
          for (i <- 1 until eventSize) args += record.get(i)
        }
        in.close()
        processEvent(name, args.toList)
      }
    }
  }

  private def submitTimedCSVFromLines(lines: List[String]): Unit = {
    var pendingEvents: List[(String, List[Any])] = Nil
    var currentTimestamp: Int = -1
    var lastLineNrInBatch: Int = 0

    for (line <- lines) {
      lineNr += 1
      val trimmed = line.trim
      if (trimmed.nonEmpty) {
        printProgress()
        if (!trimmed.startsWith("@")) {
          println(s"*** Warning: line $lineNr in #format: timed file does not start with @: $trimmed")
        }
        val spaceIdx = trimmed.indexOf(' ')
        assert(spaceIdx > 1, s"Line $lineNr: expected '@<timestamp> event,...' but got: $trimmed")
        val timeStamp = trimmed.substring(1, spaceIdx).toInt
        val eventPart = trimmed.substring(spaceIdx + 1)

        val in: Reader = new java.io.StringReader(eventPart)
        val record = CSVFormat.DEFAULT.parse(in).asScala.head
        val name = record.get(0)
        var args = new ListBuffer[Any]()
        val eventSize = record.size()
        if (eventSize > 1 && record.get(1).startsWith("[")) {
          args += (1 until eventSize).map(record.get).toList.mkString(", ")
        } else {
          for (i <- 1 until eventSize) args += record.get(i)
        }
        in.close()

        if (timeStamp != currentTimestamp && pendingEvents.nonEmpty) {
          val savedLineNr = lineNr
          lineNr = lastLineNrInBatch
          setTime(currentTimestamp)
          submitState(pendingEvents)
          lineNr = savedLineNr
          pendingEvents = Nil
        }
        currentTimestamp = timeStamp
        lastLineNrInBatch = lineNr
        pendingEvents = pendingEvents :+ (name, args.toList)
      }
    }
    if (pendingEvents.nonEmpty) {
      lineNr = lastLineNrInBatch
      setTime(currentTimestamp)
      submitState(pendingEvents)
    }
  }

  private def submitGroupedCSVFromLines(lines: List[String]): Unit = {
    for (line <- lines) {
      lineNr += 1
      val trimmed = line.trim
      if (trimmed.nonEmpty) {
        printProgress()
        val segments = trimmed.split("\\s*\\|\\s*")
        if (segments.length == 1) {
          // Single event — process normally
          val in: Reader = new java.io.StringReader(trimmed)
          val record = CSVFormat.DEFAULT.parse(in).asScala.head
          val name = record.get(0)
          var args = new ListBuffer[Any]()
          val eventSize = record.size()
          if (eventSize > 1 && record.get(1).startsWith("[")) {
            args += (1 until eventSize).map(record.get).toList.mkString(", ")
          } else {
            for (i <- 1 until eventSize) args += record.get(i)
          }
          in.close()
          processEvent(name, args.toList)
        } else {
          // Multiple events — composite state
          var events: List[(String, List[Any])] = Nil
          for (segment <- segments) {
            val seg = segment.trim
            if (seg.nonEmpty) {
              val in: Reader = new java.io.StringReader(seg)
              val record = CSVFormat.DEFAULT.parse(in).asScala.head
              val name = record.get(0)
              var args = new ListBuffer[Any]()
              val eventSize = record.size()
              if (eventSize > 1 && record.get(1).startsWith("[")) {
                args += (1 until eventSize).map(record.get).toList.mkString(", ")
              } else {
                for (i <- 1 until eventSize) args += record.get(i)
              }
              in.close()
              events = events :+ (name, args.toList)
            }
          }
          if (events.nonEmpty) {
            submitState(events)
          }
        }
      }
    }
  }

  private def processEvent(name: String, args: List[Any]): Unit = {
    if (Options.PRE_PREDICTION && preMonitor != null) {
      val modifiedEvent = preMonitor.evaluate(name, args: _*)
      modifiedEvent match {
        case Some(first :: second :: _) =>
          submit(first.toString, second.asInstanceOf[List[String]])
        case Some(event_name: String) =>
          if (event_name != "skip") submit(event_name.toString, Nil)
        case Some(_) =>
          println("Unexpected event structure output from the pre processing")
        case None =>
          submit(name, args)
      }
    } else {
      submit(name, args)
    }
  }

  private def printProgress(): Unit = {}

  /**
    * Called at the end of a trace analysis. Only called in connection of
    * log analysis (analysis of finite traces).
    */

  def end(): Unit = {
    println(s"\n$errors errors detected!\n")
    if (Options.STATISTICS) println(statistics)
    if (garbageWasCollected) {
      println("*** GARBAGE COLLECTOR WAS ACTIVATED!")
    } else {
      println("- Garbage collector was not activated")
    }
  }

  /**
    * Evaluates all formulas on a new state (new event). In case a property is violated an
    * error message is printed. There is currently no other consequence of a violated
    * property.
    */

  def evaluate(): Map[String, Boolean] = {
    debug(s"\ncurrentTime = $currentTime\n$state\n")

    formulae.map { formula =>
      formula.setTime(deltaTime)
      val result = formula.evaluate()

      if (!result) {
        errors += 1

        if (Options.PRINTS_STAT) {
          println(s"\n*** Property ${formula.name} violated on event number $lineNr:\n")
          println(state)
        }
      }

      formula.name -> result
    }.toMap
  }

  /**
    * Records property violation in the result file. Currently only event number
    * of violating event is recorded. This information is used for unit testing.
    */

  def recordResult(): Unit = {
    writelnResult(lineNr)
  }

  /**
    * Prints information useful for understanding the data written to the profile CSV file.
    */

  def printProfileHeader(): Unit = {
    formulae(0).printProfileHeader()
  }
}

/**
  * Every formula will be defined as a class extending this class.
  *
  */

abstract class Formula(val monitor: Monitor) {
  // A property named xyz will be defined by a class Formula_xyz. Pick out the name xyz:
  var name: String = this.getClass.getSimpleName.stripPrefix("Formula_")
  // BDD generator:
  var bddGenerator: BDDGenerator = null
  // Pre and now arrays, as in article:
  var pre: Array[BDD] = null
  var now: Array[BDD] = null
  // temporary pointer, used to swap the pre and now arrays:
  var tmp: Array[BDD] = null
  // maps sub-formula indexes to the text format of the sub-formulas, used for
  // debugging purposes:
  var txt: Array[String] = null
  // indices of temporal formulas, used for computing free assignments during garbage collection:
  val indices: List[Int]
  // stores variable-value-bdd pairs of newly detected values for most recent event, null means no relations:
  val emptyTouchedSet: Set[(String, Any, BDD)] = Set()
  var touchedByLastEvent: Set[(String, Any, BDD)] = emptyTouchedSet
  // records variables referred to in relations. Used to pre-condition update of above variable:
  var varsInRelations: Set[String] = Set()

  /**
    * Type of relational operators.
    */

  trait RelOp {
    def compare(v1: Any, v2: Any): Boolean
  }

  /**
    * The '<' relational operator.
    */

  case object LTOP extends RelOp {
    def compare(v1: Any, v2: Any): Boolean = {
      v1.asInstanceOf[String].toInt < v2.asInstanceOf[String].toInt
    }

    override def toString = "<"
  }

  /**
    * The '<=' relational operator.
    */

  case object LEOP extends RelOp {
    def compare(v1: Any, v2: Any): Boolean = {
      v1.asInstanceOf[String].toInt <= v2.asInstanceOf[String].toInt
    }

    override def toString = "<="
  }

  /**
    * The '>' relational operator.
    */

  case object GTOP extends RelOp {
    def compare(v1: Any, v2: Any): Boolean = {
      v1.asInstanceOf[String].toInt > v2.asInstanceOf[String].toInt
    }

    override def toString = ">"
  }

  /**
    * The '>=' relational operator.
    */

  case object GEOP extends RelOp {
    def compare(v1: Any, v2: Any): Boolean = {
      v1.asInstanceOf[String].toInt >= v2.asInstanceOf[String].toInt
    }

    override def toString = ">="
  }

  /**
    * The '=' relational operator.
    */

  case object EQOP extends RelOp {
    def compare(v1: Any, v2: Any): Boolean = {
      v1 == v2
    }

    override def toString = "="
  }

  /**
    * Turns an optional binding from variable names to values (an assignment) into a BDD.
    * This is achieved by computing the BDD for each variable/value pair and the  AND-ing these BDDs
    * together. The function is called when an event pattern has matched an incoming
    * event in the state.
    *
    * @param binding the binding to convert into a BDD.
    * @return the BDD resulting from and-ing the BDDs for each variable binding in <code>binding</code>.
    */

  def bddFromBinding(binding: Option[Binding]): BDD = {
    binding match {
      case None => bddGenerator.False
      case Some(b) =>
        var bdd: BDD = bddGenerator.True
        for ((x, v) <- b) {
          bdd = bdd.and(bddGenerator.getBddOf(x, v))
        }
        bdd
    }
  }

  /**
    * Converts a list of bindings (from matching across multiple events in a composite state)
    * into a single BDD by OR-ing together the BDD for each individual binding.
    *
    * @param bindings the list of bindings from matching against all events in the composite state.
    * @return the BDD representing the disjunction of all matching bindings.
    */

  def bddFromBindings(bindings: List[Binding]): BDD = {
    if (bindings.isEmpty) bddGenerator.False
    else bindings.map(b => bddFromBinding(Some(b))).reduce(_.or(_))
  }

  /**
    * Builds a BDD from an event pattern, matching it against the latest
    * incoming event in the current state. A particular event pattern either matches the
    * current event or not. If so, values are bound to formal parameter names of the event,
    * forming a binding (assignment). The BDD is then created from this binding.
    *
    * @param name     the name of the event.
    * @param patterns the patterns that are meant to match the arguments of the actual event.
    * @return the BDD resulting from the match, False if no match occurred.
    */

  def build(name: String)(patterns: Pattern*): BDD =
    bddFromBindings(monitor.state.holdsAny(name, patterns.toList))

  /**
    * Builds a BDD from a relational expression of the form: <code>varName1 op varName2</code>, only
    * comparing the new values seen in this event.
    *
    * @param varName1 the name of the left-hand side variable.
    * @param op       the operator.
    * @param varName2 the name of the right-hand side variable.
    * @return the BDD resulting from comparing new values against previous values.
    */

  def relation(varName1: String, op: RelOp, varName2: String): BDD = {
    val variable1 = bddGenerator.varMap(varName1)
    val variable2 = bddGenerator.varMap(varName2)
    var result: BDD = bddGenerator.False
    for ((varName, value, bdd) <- touchedByLastEvent) {
      if (varName == varName1) {
        for ((value2, bdd2) <- variable2.bdds) {
          if (op.compare(value, value2)) {
            result = result.or(bdd.and(bdd2))
            variable1.inRelation(bdd)
            variable2.inRelation(bdd2)
            debug(s"adding [$varName1:$value] $op $varName2:$value2 to '$varName1 $op $varName2'  BDD")
          }
        }
      }
      if (varName == varName2) {
        for ((value1, bdd1) <- variable1.bdds) {
          if (op.compare(value1, value)) {
            result = result.or(bdd.and(bdd1))
            variable1.inRelation(bdd1)
            variable2.inRelation(bdd)
            debug(s"adding $varName1:$value1 $op [$varName2:$value] to '$varName1 $op $varName2'  BDD")
          }
        }
      }
    }
    result
  }

  /**
    * Builds a BDD from a relational expression of the form: <code>varName op const</code>, only
    * comparing the new values seen in this event.
    *
    * @param varName the name of the left-hand side variable.
    * @param op      the operator.
    * @param const   the right-hand side constant.
    * @return the BDD resulting from comparing new value against the constant.
    */

  def relationToConstant(varName: String, op: RelOp, const: Any): BDD = {
    val variable = bddGenerator.varMap(varName)
    var result: BDD = bddGenerator.False
    for (case (`varName`, value, bdd) <- touchedByLastEvent if op.compare(value, const)) {
      result = result.or(bdd)
      variable.inRelation(bdd)
      debug(s"adding [$varName:$value] $op $const to '$varName $op $const'  BDD")
    }
    result
  }

  /**
    * If the formula contains relations (<code>touchedByLastEvent != null</code>), this function
    * adds a newly generated binding of a variable to a value and binding. This is used for updating
    * the relational expressions.
    *
    * @param name  the name of the variable.
    * @param value the value it is bound to.
    * @param bdd   the corresponding BDD generated.
    */

  def addTouchedByLastEvent(name: String, value: Any, bdd: BDD): Unit = {
    if (varsInRelations.contains(name)) {
      touchedByLastEvent += ((name, value, bdd))
      debug(s"recording binding $name -> $value (assignment: ${bdd.satOne()}) for subsequent relation updating")

    }
  }

  /**
    * Adds a time value `d` to a time value `t`, resulting in the new time value `u` using
    * carrier bits `c` as auxiliary variables.
    *
    * Time values are represented by BDDs. Each such BDD, call it  `B`, represents a sequence of
    * bits `B1,...,Bn` mentioned from least significant bit to most significant bit. This
    * allows a recursive algorithm, which adds bits from lowest to highest significant bit.
    * The `c` the carrier bits used to carry over.
    *
    * E.g. say we want to add `t=01`` (the number 1) and `d=01` (the number 1). These are
    * passed to this function as `10` and `10` respectively (least significant bits first).
    * The function adds `1` and `1` giving `0` and resulting in carrier bit `c1` being `1`.
    * `c1=1` is then used when adding the two `0` resulting in `1`, overall resulting in `01`
    * with the least significant but mentioned first, hence this is
    * `10` in normal bit format (the number 2).
    *
    * The function takes care of the first (least significant) bit addition, and then calls
    * `addConstRest` for the rest of the bits.
    *
    * @param t the time value to add to (from previous event).
    * @param u the resulting time value.
    * @param d the time delta to add to `t` (the time difference between this and previous event).
    * @param c the carrier bits used as auxiliary variable.
    * @return the BDD defining the result `u` of the addition.
    **/

  def addConst(t: List[BDD], u: List[BDD], d: List[BDD], c: List[BDD]): BDD = {
    (t, u, d, c) match {
      case (t_bit :: t_rest, u_bit :: u_rest, d_bit :: d_rest, c_bit :: c_rest) =>
        val initBDD = u_bit.biimp(t_bit.xor(d_bit))
        val initCarrier = c_bit.biimp(t_bit.and(d_bit))
        initBDD.and(initCarrier).and(addConstRest(t_rest, u_rest, d_rest, c_bit :: c_rest))
      case _ => assert(false, "addConst pattern match fails").asInstanceOf[BDD]
    }
  }

  /**
    * Adds a time value `d` to a time value `t`, resulting in the new time value `u` using
    * carrier bits `c` as auxiliary variables.
    *
    * This function is called on all bits following the least significant bit. See
    * `addConst`.
    *
    * @param t the time value to add to (from previous event).
    * @param u the resulting time value.
    * @param d the time delta to add to `t` (the time difference between this and previous event).
    * @param c the carrier bits used as auxiliary variable.
    * @return the BDD defining the result `u` of the addition.
    */

  def addConstRest(t: List[BDD], u: List[BDD], d: List[BDD], c: List[BDD]): BDD = {
    (t, u, d, c) match {
      case (Nil, Nil, Nil, _) => bddGenerator.True
      case (t_bit :: t_rest, u_bit :: u_rest, d_bit :: d_rest, c_prev :: c_cur :: c_rest) =>
        val u_bit_def = u_bit.biimp(t_bit.xor(d_bit).xor(c_prev))
        val c_cur_def = c_cur.biimp(
          (t_bit.and(d_bit)).or(
            t_bit.and(c_prev).or(
              d_bit.and(c_prev)
            )
          ))
        u_bit_def.and(c_cur_def.and(addConstRest(t_rest, u_rest, d_rest, c_cur :: c_rest)))
      case _ => assert(false, "addConstRest pattern match fails").asInstanceOf[BDD]
    }
  }

  /**
    * Returns true of the first `bit1` is one and the second `bit2` is zero.
    *
    * @param bit1 the first bit.
    * @param bit2 the second bit.
    * @return the `True` BDD if the first bit is one and the second is zero.
    */

  def gtBit(bit1: BDD, bit2: BDD): BDD =
    bit1.and(bit2.not())

  /**
    * Determines whether one time value `u` is strictly bigger than another `l`.
    * The time values are presented with the most significant bit first, and the
    * function recurses over the bits comparing them until the result becomes
    * obvious.
    *
    * E.g. to compare binary `101` (number 5) to binary `110` (number 6) the
    * function first compares the first two `1`s, which does not determine the
    * result. It then moves on to the next two bits `0` and `1`, and here it becomes
    * clear that the first number is not bigger than the second.
    *
    * @param u the first time value (the time difference of the current event)
    * @param l the second time value (the limit constant associated with the S-operator)
    * @return the result of the comparison, `True` if the first number is bigger than the second.
    *         Otherwise `False`.
    */

  def gtConst(u: List[BDD], l: List[BDD]): BDD = {
    (u, l) match {
      case (Nil, Nil) => bddGenerator.False
      case (u_bit :: u_rest, l_bit :: l_rest) =>
        gtBit(u_bit, l_bit).ite(
          bddGenerator.True,
          gtBit(l_bit, u_bit).ite(
            bddGenerator.False,
            gtConst(u_rest, l_rest)
          )
        )
      case _ => assert(false, "gtConst pattern match fails").asInstanceOf[BDD]
    }
  }

  /**
    * From a list of BDD variable-numbers (the JavaBDD package represents a
    * variable by a number), the function returns a list of the BDDs, one for
    * each of these variables. The BDD returns `1` for `1` and `0` for `0`.
    *
    * @param positions the numbers of the variables.
    * @return the corresponding one-bit BDDs.
    */

  def generateBDDList(positions: Array[Int]): List[BDD] = {
    for (pos <- positions.toList) yield bddGenerator.theOneBDDFor(pos)
  }

  /**
    * Sets the time delta in the individual formula. Note that the delta stored in the
    * individual formula is a function of the maximal time limit occurring in
    * the formula, in order to save bits. It is meant to be overridden by each
    * formula class if the formula contains time constraints.
    *
    * @param actualDelta the actual difference in time between the timestamp of
    *                    the previous event and the current event.
    */

  def setTime(actualDelta: Int): Unit = {}

  /**
    * Returns the True BDD if the Delta time (the time difference between the
    * time stamp of the current event and the time stamp of the previous event)
    * is less than the time limit passed as argument, otherwise the False BDD
    * is returned.
    *
    * @param timeLimit the timelimit (small delta) occuring as part of a temporal
    *                  operator in the property being evaluated.
    * @return the True or False BDD, depending on whether the time passed since last
    *         event is less than the argument time value or not.
    */

  def deltaLessThanTimeLimit(timeLimit: Int): BDD = {
    if (monitor.deltaTime < timeLimit)
      bddGenerator.True
    else
      bddGenerator.False
  }

  /**
    * Declares all variables (each identified by a name) in a formula.
    * This includes initializing the BDD generator, which is stored in
    * <code>bddGenerator</code>, and initializing <code>True</code> and
    * <code>False</code>. The result returned is a list of the Variable objects.
    * In addition BDD variables are allocated for keeping track of time in case
    * the property contains timed temporal operators. In this case `bitsPerTimeVar > 0`.
    *
    * @param variables      the (name,bounded) pairs for variables in a formula.
    * @param bitsPerTimeVar the number of bits to be allocated per time variable.
    *                       This number is `0` if the property does not contain
    *                       timed temporal operators.
    * @return a list of Variable objects, one for each variable.
    */

  def declareVariables(variables: (String, Boolean)*)(bitsPerTimeVar: Int): List[Variable] = {
    val variableList = variables.toList
    val nameList: List[String] = variableList.map(_._1)
    val varsAndBitsPerVar = variableList.map {
      case (n, b) => (n, b, Options.BITS)
    }
    bddGenerator = new BDDGenerator(this)(varsAndBitsPerVar, bitsPerTimeVar)
    bddGenerator.initializeVariables()
    nameList.map(bddGenerator.varMap(_))
  }

  /**
    * The evaluation method for a formula. Must be overridden for each formula.
    * The method will evaluate the formula on each new event.
    *
    * @return true iff. the formula is true on the trace seen so far.
    */

  def evaluate(): Boolean

  /**
    * Returns a string representation of the current values of the <code>pre</code> and
    * <code>now</code> arrays. For each index into these arrays also the text of the
    * subformula is printed for better comprehension.
    *
    * @return string representation of formula state.
    */

  override def toString: String = {
    var result: String = ""
    result += s"===============\n"
    result += s"Property $name:\n"
    result += s"===============\n"
    for (i <- 0 to now.size - 1) {
      result += s"[$i] ${txt(i)}\n\n"
      result += s"pre: ${bddToString(pre(i))}\n"
      result += s"now: ${bddToString(now(i))}\n"
      result += s"-------------\n"
    }
    result
  }

  /**
    * Prints information useful for understanding what is written to the profile CSV file.
    */

  def printProfileHeader(): Unit = {
    println()
    println("================")
    println(s"Property: $name")
    println("================")
    println()
    println("Profile data written to CSV file dejavu-profile.csv")
    println()
    println("Formulas:")
    println()
    val profiledIndices = indices
    for (i <- profiledIndices) {
      println(s"----- $i -----")
      println(txt(i))
      writeProfile(s"nodeCount$i,pathCount$i,satCount$i,compr$i,")
    }
    writelnProfile()
  }

  /**
    * Prints a formula state for debugging. This includes whether the formula is true or not,
    * and the value of the <code>now</code> array, where each entry is printed both as a one
    * line text value, and also as a graph in dot format for visualization with GraphViz.
    *
    * In profile mode, profiling data are written to a CSV file.
    */

  def debugMonitorState(): Unit = {
    if (Options.PROFILE) { // Designed to profile one formula.
      val line = new StringBuffer()
      val profiledIndices = indices
      for (i <- profiledIndices) {
        val bdd: BDD = now(i)
        val nodeCount: Int = bdd.nodeCount()
        val pathCount: Double = bdd.pathCount()
        val satCount: Double = bdd.satCount()
        val compression: Double = if (nodeCount != 0) satCount / nodeCount else 0
        line.append(s"$nodeCount,$pathCount,$satCount,$compression,")
      }
      writelnProfile(line)
    }
    if (Options.DEBUG) {
      println("================")
      println(s"Property: $name")
      println("================")
      println()
      if (now(0).isZero) {
        println("*** FALSE ***")
        println()
      }
      for (i <- now.size - 1 to 0 by -1) {
        println(s"----- $i -----")
        println(txt(i))
        if (now(i).isOne) println("TRUE") else if (now(i).isZero) println("FALSE") else {
          println(s"now:")
          println(now(i)) // prints BDD as a one line text
          now(i).printDot() // prints BDD in dot format for vizualization with GraphViz
        }
      }
    }
  }
}



  
        


/*
  prop pol_bank_transfer_balance : (forall t_1 . tx_g(t_1) -> exists b_1 . ((!(exists b2 . bal_a(b2)) S bal_a(b_1)) & !(b_1 < t_1))) & ((@ exists t_2 . tx_g(t_2)) -> exists b_2 . bal_a(b_2)) 
*/

class Formula_pol_bank_transfer_balance(monitor: Monitor) extends Formula(monitor) {
          
  override def evaluate(): Boolean = {
    // assignments1 (leaf nodes that are not rule calls):
      now(3) = build("tx_g")(V("t_1"))
      now(9) = build("bal_a")(V("b2"))
      now(10) = build("bal_a")(V("b_1"))
      now(16) = build("tx_g")(V("t_2"))
      now(18) = build("bal_a")(V("b_2"))
    // assignments2 (rule nodes excluding what is below @ and excluding leaf nodes):
    // assignments3 (rule calls):
    // assignments4 (the rest of rules that are below @ and excluding leaf nodes):
    // assignments5 (main formula excluding leaf nodes):
      now(8) = var_b2.seen.and(now(9)).exist(var_b2.quantvar)
      now(7) = now(8).not()
      now(6) = now(10).or(now(7).and(pre(6)))
      now(12) = relation("b_1",LTOP,"t_1").or(pre(12))
      now(11) = now(12).not()
      now(5) = now(6).and(now(11))
      now(4) = var_b_1.seen.and(now(5)).exist(var_b_1.quantvar)
      now(2) = now(3).not().or(now(4))
      now(1) = var_t_1.seen.imp(now(2)).forAll(var_t_1.quantvar)
      now(15) = var_t_2.seen.and(now(16)).exist(var_t_2.quantvar)
      now(14) = pre(15)
      now(17) = var_b_2.seen.and(now(18)).exist(var_b_2.quantvar)
      now(13) = now(14).not().or(now(17))
      now(0) = now(1).and(now(13))

      debugMonitorState()

      val error = now(0).isZero
      if (error) monitor.recordResult()
      tmp = now
      now = pre
      pre = tmp
      touchedByLastEvent = emptyTouchedSet

      

      !error
  }

  val var_t_1 :: var_b_1 :: var_b2 :: var_t_2 :: var_b_2 :: Nil = declareVariables(("t_1",true), ("b_1",true), ("b2",true), ("t_2",true), ("b_2",true))(0): @unchecked

  varsInRelations = Set("b_1","t_1")
  val indices: List[Int] = List(14,6)

  pre = Array.fill(19)(bddGenerator.False)
  now = Array.fill(19)(bddGenerator.False)

  txt = Array(
    "(forall t_1 . tx_g(t_1) -> exists b_1 . ((!(exists b2 . bal_a(b2)) S bal_a(b_1)) & !(b_1 < t_1))) & ((@ exists t_2 . tx_g(t_2)) -> exists b_2 . bal_a(b_2))",
      "forall t_1 . tx_g(t_1) -> exists b_1 . ((!(exists b2 . bal_a(b2)) S bal_a(b_1)) & !(b_1 < t_1))",
      "tx_g(t_1) -> exists b_1 . ((!(exists b2 . bal_a(b2)) S bal_a(b_1)) & !(b_1 < t_1))",
      "tx_g(t_1)",
      "exists b_1 . ((!(exists b2 . bal_a(b2)) S bal_a(b_1)) & !(b_1 < t_1))",
      "(!(exists b2 . bal_a(b2)) S bal_a(b_1)) & !(b_1 < t_1)",
      "!(exists b2 . bal_a(b2)) S bal_a(b_1)",
      "!(exists b2 . bal_a(b2))",
      "exists b2 . bal_a(b2)",
      "bal_a(b2)",
      "bal_a(b_1)",
      "!(b_1 < t_1)",
      "b_1 < t_1",
      "(@ exists t_2 . tx_g(t_2)) -> exists b_2 . bal_a(b_2)",
      "@ exists t_2 . tx_g(t_2)",
      "exists t_2 . tx_g(t_2)",
      "tx_g(t_2)",
      "exists b_2 . bal_a(b_2)",
      "bal_a(b_2)"
  )

  debugMonitorState()
}
        
/* The specialized Monitor for the provided properties. */

class PropertyMonitor(preMonitor: PreMonitorTrait) extends Monitor(preMonitor) {

  def eventsInSpec: Set[String] = Set("bal_a","tx_g")

  formulae ++= List(new Formula_pol_bank_transfer_balance(this))
}
      

object TraceMonitor {
  // Declare moni but don't initialize here
  private lazy val online_monitor: PropertyMonitor = new PropertyMonitor(null)

  // Usage information for the command line
  private val usage: String =
    """Usage: --logfile=<filename> [OPTIONS]
      |Options:
      |-l, --logfile=<filename>   Path to the CSV log file to be analyzed. (Optional)
      |-b, --bits=<numOfBits>     Number of bits for each variable in the BDD representation. (Default: 20 bits)
      |-m, --mode=(debug|profile) Set the output mode. (Default: None)
      |-st, --stat=(true|false)   Print violations if set to true. (Optional)
      |-c, --clear=(0|1)          Clear generated files and folders. Set to '1' for cleaning. (Optional)
      |
      |Argument Examples:
      |--logfile log.csv
      |--logfile log.csv --bits 16 --mode debug --stat false
      |""".stripMargin

  // Configuration method to initialize the monitor
  def config(bits: String, mode: String, printStat: String, resultFile: String = "output/resultFile"): Boolean = {
    // Validate and set 'bits'
    if (!bits.trim.matches("""\d+""")) {
      println("Error: 'bits' argument must be an integer.")
      return false
    }
    Options.BITS = bits.trim.toInt

    // Initialize 'online_monitor' after setting 'bits'
    // 'online_monitor' will be initialized lazily when accessed

    // Set the 'mode' option
    mode.toLowerCase() match {
      case "debug" => Options.DEBUG = true
      case "profile" =>
        Options.PROFILE = true
        openProfileFile("dejavu-profile.csv")
        online_monitor.printProfileHeader()
      case _ => // Default case, do nothing or log if needed
    }

    // Set the 'printStat' option
    printStat.toLowerCase() match {
      case "true" => Options.PRINTS_STAT = true
      case "false" => Options.PRINTS_STAT = false
      case _ =>
        println("Error: 'stat' argument must be 'true' or 'false'.")
        return false
    }

    // Configure the result file path and create directories if necessary
    Options.RESULT_FILE = resultFile
    val dir = new File(Options.RESULT_FILE)
    val parentDir = dir.getParentFile
    if (parentDir != null && !parentDir.exists() && !parentDir.mkdirs()) {
      println(s"Error: Failed to create the parent directory: ${parentDir.getAbsolutePath}")
      return false
    }

    openResultFile(Options.RESULT_FILE)

    println(s"Configuration complete: Bits=$bits, Mode=$mode, PrintStat=$printStat, ResultFile=$resultFile")
    true
  }

  // Method to evaluate events
  def eval(event: String): String = {
    val input = event.split(",")
    val name = input.headOption.getOrElse("")
    val args = if (input.length > 1 && input(1).startsWith("[")) {
      List(input.tail.mkString(", "))
    } else {
      input.tail.toList
    }

    val res: String = name match {
      case "#init#" =>
        online_monitor.formulae.map { formula =>
          s"${formula.name}=false"
        }.mkString(",")
      case "#skip#" =>
        online_monitor.end()
        "#skip#=false"
      case "#end#" =>
        online_monitor.end()
        "#end#=true"
      case _ =>
        online_monitor.lineNr += 1
        val resultMap: Map[String, Boolean] = online_monitor.submit(name, args)
        resultMap.map { case (key, value) => s"$key=$value" }.mkString(",")
    }

    res
  }

  // Process multiple events as one atomic time step (composite event).
  // Format: "name1,arg1,arg2|name2,arg1|name3" (pipe-delimited events)
  def processComposite(events: String): String = {
    val sep = "[|]"
    val eventList = events.split(sep).toList.map { eventStr =>
      val parts = eventStr.split(",")
      val name = parts.headOption.getOrElse("")
      val args: List[Any] = if (parts.length > 1) parts.tail.toList else Nil
      (name, args)
    }
    if (eventList.isEmpty) return ""
    online_monitor.lineNr += 1
    val resultMap: Map[String, Boolean] = online_monitor.submitState(eventList)
    resultMap.map { case (key, value) => s"$key=$value" }.mkString(",")
  }

  def end_eval(): Unit = {
    closeResultFile()
  }

  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Evaluation time: " + (t1 - t0) / 1e9d + "s")
    result
  }

  def main(args: Array[String]): Unit = {

    if (2 <= args.length && args.length <= 10 && args.length % 2 == 0) {
      val argMapBuilder = Map.newBuilder[String, Any]
      args.sliding(2, 2).toList.collect {
        case Array("--logfile", logfile: String) => argMapBuilder.+=("logfile" -> logfile)
        case Array("--bits", numOfBits: String) => argMapBuilder.+=("bits" -> numOfBits)
        case Array("--mode", mode: String) => argMapBuilder.+=("mode" -> mode)
        case Array("--resultfile", resultfile: String) => argMapBuilder.+=("resultfile" -> resultfile)
        case Array("--stat", stat: String) => argMapBuilder.+=("stat" -> stat)
      }

      val argMap = argMapBuilder.result()

      val logFile = argMap.get("logfile")
      val logfilePath = logFile match {
        case Some(value) => value.toString
        case None =>
          println(s"*** program must be called with logfile argument")
          println(usage)
          return
      }

      var dir = new File(logfilePath)
      if (!dir.exists) {
        println(s" ***logfile is not a valid file")
        return
      }

      val resultfile = argMap.get("resultfile")
      Options.RESULT_FILE = resultfile match {
        case Some(value) => value.toString
        case None => "/Users/moraneus/Downloads/dejavu/LLMrv/dejavuguard/scenario_runner/scenarios/banking_scenario/./src/test/scala/sandbox/generated_monitors/test_nQUQNyN2aX/dejavu-results"
      }

      dir = new File(Options.RESULT_FILE)
      if (!dir.getParentFile.exists) {
        println(s" ***resultfile parent is not a valid folder")
        return
      }

      val bits = argMap.get("bits")
      val bitsValue = bits match {
        case Some(value) =>
          if (!value.toString.matches("""\d+""")) {
            println(s"*** bits argument must be an integer")
            return
          } else {
            value.toString
          }
        case None => "20" // Default is 20 bits length
      }
      Options.BITS = bitsValue.toInt

      val mode = argMap.get("mode")
      mode match {
        case Some(value) =>
          val modeValue = value.toString.toLowerCase()
          if (modeValue == "debug") Options.DEBUG = true
          else if (modeValue == "profile") Options.PROFILE = true
          else {
            println(s"*** mode argument must be: debug or profile")
            return
          }
        case None => println("No mode was selected")
      }

      val printStat = argMap.get("stat")
      printStat match {
        case Some(value) =>
          val printStatValue = value.toString.toLowerCase()
          if (printStatValue == "true") Options.PRINTS_STAT = true
          else if (printStatValue == "false") Options.PRINTS_STAT = false
          else {
            println(s"*** stat argument must be: true or false")
            return
          }
        case None => println("Default for stat is true")
      }

      val m = new PropertyMonitor(null)

      try {
      time {
        openResultFile(Options.RESULT_FILE)
        if (Options.PROFILE) {
          openProfileFile("dejavu-profile.csv")
          m.printProfileHeader()
        }
        m.submitCSVFile(logfilePath)
       }
         

     } catch {
        case e: Throwable =>
          println(s"\n*** $e\n")
        // e.printStackTrace()
      } finally {
        closeResultFile()
        if (Options.PROFILE) closeProfileFile()
      }
    } else {
      println("*** call with these arguments:")
      println(usage)
    }
  }
}
      
