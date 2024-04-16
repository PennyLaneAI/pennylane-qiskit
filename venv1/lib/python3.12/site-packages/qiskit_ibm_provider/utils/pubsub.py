# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Message broker for the Publisher / Subscriber mechanism
"""

from __future__ import annotations

import typing

from qiskit.exceptions import QiskitError

try:
    from qiskit.tools.events.pubsub import _Broker as _QiskitBroker
except ImportError:
    _QiskitBroker = None

_Callback = typing.Callable[..., None]


class _Broker:
    """The event/message broker. It's a singleton.

    In order to keep consistency across all the components, it would be great to
    have a specific format for new events, documenting their usage.
    It's the responsibility of the component emitting an event to document it's usage in
    the component docstring.

    Event format::

        "<namespace>.<component>.<action>"

    Examples:

    * "ibm.job.start"
    """

    _instance: _Broker | None = None
    _subscribers: dict[str, list[_Subscription]] = {}

    @staticmethod
    def __new__(cls: type[_Broker]) -> _Broker:
        if _Broker._instance is None:
            # Backwards compatibility for Qiskit pre-1.0; if the Qiskit-internal broker
            # singleton exists then we use that instead of defining a new one, so that
            # the event streams will be unified even if someone is still using the
            # Qiskit entry points to subscribe.
            #
            # This dynamic switch assumes that the interface of this vendored `Broker`
            # code remains identical to the Qiskit 0.45 version.
            _Broker._instance = object.__new__(_QiskitBroker or cls)
        return _Broker._instance

    class _Subscription:
        def __init__(self, event: str, callback: _Callback):
            self.event: str = event
            self.callback: _Callback = callback

        def __eq__(self, other: object) -> bool:
            """Overrides the default implementation"""
            if isinstance(other, self.__class__):
                return self.event == other.event and id(self.callback) == id(
                    other.callback
                )  # Allow 1:N subscribers
            return False

    def subscribe(self, event: str, callback: _Callback) -> bool:
        """Subscribes to an event, so when it's emitted all the callbacks subscribed,
        will be executed. We are not allowing double registration.

        Args:
            event (string): The event to subscribed in the form of:
                            "terra.<component>.<method>.<action>"
            callback (callable): The callback that will be executed when an event is
                                  emitted.
        """
        if not callable(callback):
            raise QiskitError("Callback is not a callable!")

        if event not in self._subscribers:
            self._subscribers[event] = []

        new_subscription = self._Subscription(event, callback)
        if new_subscription in self._subscribers[event]:
            # We are not allowing double subscription
            return False

        self._subscribers[event].append(new_subscription)
        return True

    def dispatch(self, event: str, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Emits an event if there are any subscribers.

        Args:
            event (String): The event to be emitted
            args: Arguments linked with the event
            kwargs: Named arguments linked with the event
        """
        # No event, no subscribers.
        if event not in self._subscribers:
            return

        for subscriber in self._subscribers[event]:
            subscriber.callback(*args, **kwargs)

    def unsubscribe(self, event: str, callback: _Callback) -> bool:
        """Unsubscribe the specific callback to the event.

        Args
            event (String): The event to unsubscribe
            callback (callable): The callback that won't be executed anymore

        Returns
            True: if we have successfully unsubscribed to the event
            False: if there's no callback previously registered
        """

        try:
            self._subscribers[event].remove(self._Subscription(event, callback))
        except KeyError:
            return False

        return True

    def clear(self) -> None:
        """Unsubscribe everything, leaving the Broker without subscribers/events."""
        self._subscribers.clear()


class Publisher:
    """Represents a "publisher".

    Every component (class) can become a :class:`Publisher` and send events by
    inheriting this class. Functions can call this class like::

        Publisher().publish("event", args, ... )
    """

    def __init__(self) -> None:
        self._broker: _Broker = _Broker()

    def publish(self, event: str, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Triggers an event, and associates some data to it, so if there are any
        subscribers, their callback will be called synchronously."""
        return self._broker.dispatch(event, *args, **kwargs)


class Subscriber:
    """Represents a "subscriber".

    Every component (class) can become a :class:`Subscriber` and subscribe to events,
    that will call callback functions when they are emitted.
    """

    def __init__(self) -> None:
        self._broker: _Broker = _Broker()

    def subscribe(self, event: str, callback: _Callback) -> bool:
        """Subscribes to an event, associating a callback function to that event, so
        when the event occurs, the callback will be called.

        This is a blocking call, so try to keep callbacks as lightweight as possible."""
        return self._broker.subscribe(event, callback)

    def unsubscribe(self, event: str, callback: _Callback) -> bool:
        """Unsubscribe a pair event-callback, so the callback will not be called anymore
        when the event occurs."""
        return self._broker.unsubscribe(event, callback)

    def clear(self) -> None:
        """Unsubscribe everything"""
        self._broker.clear()
