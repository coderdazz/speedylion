import win32com.client as win32


class EmailScraper:
    """Email Scraper to search for emails in Outlook Inbox"""

    def __init__(self):
        self._outlook = win32.Dispatch('outlook.application')
        mapi = self._outlook.GetNamespace("MAPI")
        self._messages = mapi.GetDefaultFolder(6).Items
        self._messages.Sort("[ReceivedTime]", True)

    def find_by_title(self, message_title: str, search_limit: int = 100, multiple=False):
        """Find message by title"""

        print(f'Searching for {message_title} in Outlook Inbox...')

        found_messages = []

        for count, message in enumerate(self._messages):
            if count > search_limit:
                print(f'Search limit of {search_limit} reached!')
                break
            if message_title in message.subject:
                print(message.subject)
                print(message.ReceivedTime)
                print("-------------------------")
                if multiple:
                    found_messages.append(message)
                else:
                    return message

        if not found_messages:
            print('No messages found!')
            return None

        return found_messages

    def send_attachment(self, filename, recipient, title):
        """Send attachment to recipient"""

        msg = self._outlook.CreateItem(0)
        msg.Subject = title
        msg.To = recipient
        msg.Attachments.Add(filename)
        msg.Send()

        print('Message sent')
