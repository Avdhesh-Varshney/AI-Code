# pip install phonenumbers

import phonenumbers
from phonenumbers import geocoder, carrier, timezone

phone = input("Enter Phone Number with Country Code Here: ")

phone_number = phonenumbers.parse(phone)

# Here is how to know the country name
country_name = geocoder.description_for_number(phone_number, 'en')
print("Country Name:", country_name)

# To know about the Service Provider
service_provider = carrier.name_for_number(phone_number, 'en')
print("Service Provider:", service_provider)

# To know the time zone
time_zone = timezone.time_zones_for_number(phone_number)
print("Time Zone: ", time_zone)
