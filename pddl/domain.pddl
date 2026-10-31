; Warehouse robots: move through the aisles, pick items from shelves, drop them at stations.
; A location holds at most one robot, so plans with several robots must avoid collisions.
; A robot carries at most one item.
(define (domain warehouse)
  (:requirements :strips :typing)
  (:types
    robot item location - object
    aisle shelf station - location)
  (:predicates
    (at ?r - robot ?l - location)
    (connected ?from ?to - location)
    (clear ?l - location)
    (stored ?i - item ?s - shelf)
    (holding ?r - robot ?i - item)
    (empty ?r - robot)
    (delivered ?i - item ?s - station))

  (:action move
    :parameters (?r - robot ?from ?to - location)
    :precondition (and (at ?r ?from) (connected ?from ?to) (clear ?to))
    :effect (and (at ?r ?to) (clear ?from) (not (at ?r ?from)) (not (clear ?to))))

  (:action pick
    :parameters (?r - robot ?i - item ?s - shelf)
    :precondition (and (at ?r ?s) (stored ?i ?s) (empty ?r))
    :effect (and (holding ?r ?i) (not (stored ?i ?s)) (not (empty ?r))))

  (:action drop
    :parameters (?r - robot ?i - item ?s - station)
    :precondition (and (at ?r ?s) (holding ?r ?i))
    :effect (and (delivered ?i ?s) (empty ?r) (not (holding ?r ?i)))))
