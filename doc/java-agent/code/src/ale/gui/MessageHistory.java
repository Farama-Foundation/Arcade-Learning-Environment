/*
 * Java Arcade Learning Environment (A.L.E) Agent
 *  Copyright (C) 2011-2012 Marc G. Bellemare <mgbellemare@ualberta.ca>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package ale.gui;

import java.util.LinkedList;
import java.util.List;

/** Encapsulates a list of messages. Each message is timestamped.
 *
 * @author Marc G. Bellemare <mgbellemare@ualberta.ca>
 */
public class MessageHistory {
    public class Message {
        protected String text;
        protected long timeStamp;

        public Message(String text, long timeStamp) {
            this.text = text;
            this.timeStamp = timeStamp;
        }

        public String getText() { return text; }
        public long getTimeStamp() { return timeStamp; }
    }

    /** A list of messages, with the first element being the oldest */
    protected LinkedList<Message> messages;

    public MessageHistory() {
        messages = new LinkedList<Message>();
    }
    
    /** Adds a message to our history. The time at which the message was added
     *   is also recorded.
     *
     * @param text The message to be added.
     */
    public void addMessage(String text) {
        long currentTime = System.currentTimeMillis();

        messages.addLast(new Message(text, currentTime));
    }

    /** Returns a list of current messages */
    public List<Message> getMessages() {
        return messages;
    }

    /** Remove any message which is older than 'maxAge'. The age of a message is
     *   found by comparing its timestamp with the current time.
     *
     * @param maxAge The maximum age, in milliseconds, of a message.
     */
    public void update(long maxAge) {
        long currentTime = System.currentTimeMillis();

        while (!messages.isEmpty()) {
            Message m = messages.getFirst();

            // Delete this message if it is old enough
            long age = currentTime - m.timeStamp;
            if (age > maxAge)
                messages.removeFirst();
            else
                break; // Messages are ordered by timestamp so we can stop
        }
    }
}
