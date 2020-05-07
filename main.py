import traceback
import logging
import argparse
import os

from configuration.configuration import Configuration
from datastore.mysql_datastore import MySqlDatastore
from cloudstore.dropbox_cloudstore import DropboxCloudstore
from email_app.gmail_email_app import GmailEmailApp

logger = logging.getLogger('Main')


def _setup_log(log_path: str = 'logs/output.log', debug: bool = False) -> None:
    log_path = log_path.split(os.sep)
    if len(log_path) > 1:

        try:
            os.makedirs((os.sep.join(log_path[:-1])))
        except FileExistsError:
            pass
    log_filename = os.sep.join(log_path)
    # noinspection PyArgumentList
    logging.basicConfig(level=logging.INFO if debug is not True else logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler(log_filename),
                            # logging.handlers.TimedRotatingFileHandler(log_filename, when='midnight', interval=1),
                            logging.StreamHandler()
                        ]
                        )


def _argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='A template for python projects.',
        add_help=False)
    # Required Args
    required_arguments = parser.add_argument_group('Required Arguments')
    config_file_params = {
        'type': argparse.FileType('r'),
        'required': True,
        'help': "The configuration yml file"
    }
    required_arguments.add_argument('-m', '--run-mode', choices=['run_mode_1', 'run_mode_2', 'run_mode_3'],
                                    required=True,
                                    default='run_mode_1',
                                    help='Description of the run modes')
    required_arguments.add_argument('-c', '--config-file', **config_file_params)
    required_arguments.add_argument('-l', '--log', help="Name of the output log file")
    # Optional args
    optional = parser.add_argument_group('Optional Arguments')
    optional.add_argument('-d', '--debug', action='store_true', help='Enables the debug log messages')
    optional.add_argument("-h", "--help", action="help", help="Show this help message and exit")

    return parser.parse_args()


def main():
    """
    :Example:
    python main.py -m run_mode_1
                   -c confs/template_conf.yml
                   -l logs/output.log
    """

    # Initializing
    args = _argparser()
    _setup_log(args.log, args.debug)
    logger.info("Starting in run mode: {0}".format(args.run_mode))
    # Load the configuration
    configuration = Configuration(config_src=args.config_file)
    # Init the Cloudstore
    cloud_store = DropboxCloudstore(config=configuration.get_cloudstores()[0])
    # Init the Datastore
    data_store = MySqlDatastore(**configuration.get_datastores()[0])
    # Init the Email App
    gmail_configuration = configuration.get_email_apps()[0]
    gmail_app = GmailEmailApp(config=configuration.get_email_apps()[0])

    # Mysql examples
    logger.info("\n\nMYSQL EXAMPLE\n-------------------------")
    logger.info("\n\nTables in current DB: {0}".format(list(data_store.show_tables())))
    logger.info("Creating Table: orders")
    table_schema = """ order_id INT(6) PRIMARY KEY,
                       order_type VARCHAR(30) NOT NULL,
                       location VARCHAR(30) NOT NULL """
    data_store.create_table(table='orders', schema=table_schema)
    logger.info("Tables in current DB:\n{0}".format(list(data_store.show_tables())))
    logger.info("Inserting into orders the values:\n(1 simple newyork)..")
    insert_data = {"order_id": 1,
                   "order_type": "plain",
                   "location": "new_york"}
    data_store.insert_into_table(table='orders', data=insert_data)
    logger.info("SELECT * FROM orders;\n{0}".format(data_store.select_from_table(table='orders')))
    logger.info("Deleting the inserted row from table orders..")
    data_store.delete_from_table(table='orders', where='order_id=1')
    logger.info("SELECT * FROM orders;\n{0}".format(data_store.select_from_table(table='orders')))
    logger.info("Dropping Table: orders")
    data_store.drop_table(table='orders')
    logger.info("Tables in current DB:\n{0}".format(list(data_store.show_tables())))

    # Dropbox examples
    logger.info("\n\nDROPBOX EXAMPLE\n-------------------------")
    logger.info(
        "List of files in Dropbox /python_template:\n{0}".format(list(cloud_store.ls(path='/python_template').keys())))
    upload_path = "/python_template/file1.txt"
    file_content = "test file content"
    logger.info("Uploading file {file} with content:\n{content}".format(file=upload_path, content=file_content))
    cloud_store.upload_file(file_bytes=file_content.encode(), upload_path=upload_path)
    logger.info(
        "List of files in Dropbox /python_template:\n{0}".format(list(cloud_store.ls(path='/python_template').keys())))
    downloaded_file = cloud_store.download_file(frompath=upload_path)
    logger.info("Downloaded file and its content is:\n{0}".format(downloaded_file))
    cloud_store.delete_file(file_path=upload_path)
    logger.info("Deleting file {file}..".format(file=upload_path))
    logger.info(
        "List of files in Dropbox /python_template:\n{0}".format(list(cloud_store.ls(path='/python_template').keys())))

    # Gmail examples
    logger.info("\n\nGMAIL EXAMPLE\n-------------------------")
    subject = "Email example"
    body = "<h1>This is an html body example</h1><br><b>This goes to the html argument. " \
           "You can use the text argument for plain text.</b>"
    emails_list = [gmail_configuration['email_address']]
    attachments_paths = [os.path.join('data', 'sample_data.txt')]
    logger.info(
        "Sending email with `subject` = `{subject}`, `from,to,cc,bcc,reply_to` = `{email_addr}`, "
        "`html` = `{body}` and `attachments` = `{attachments}`".format(
            subject=subject, email_addr=emails_list[0], body=body, attachments=attachments_paths))
    gmail_app.send_email(subject=subject,
                         to=emails_list,
                         cc=emails_list,
                         bcc=emails_list,
                         html=body,
                         attachments=attachments_paths,
                         sender=emails_list[0],
                         reply_to=emails_list[0]
                         )


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
